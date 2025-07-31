from lib.memory import ShortTermMemory
from lib.react_agent import ReActAgent, AgentState
from lib.state_machine import StateMachine, Step, EntryPoint, Termination
from typing import TypedDict, List, Union
import json
import logging

from lib.tooling import Tool

class GameAgentState(TypedDict):
    user_query: str
    rag_results: List[dict]
    evaluation_result: dict
    web_results: dict
    final_answer: str
    needs_web_search: bool

class GameAnswerAgent:
    def __init__(self, 
                 retrieve_game: Tool,
                 evaluate_retrieval: Tool,
                 game_web_search: Tool,
                 model_name: str = "gpt-4o-mini",                  
                 api_key: str = None, 
                 base_url: str = None,
                 log_level: str = "info",
                 session_id: str = None) -> None:
        """
        GameAnswerAgent that orchestrates RAG and Web search using state machine
        
        Workflow: UserQuery -> Retrieve RAG -> Evaluate -> Terminate or invoke WebSearch Agent
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = base_url
        self.session_id = session_id if session_id else "default_session"
        
        # Initialize memory and state machine
        self.memory = ShortTermMemory()
        
        # Instructions for the sub-agents
        rag_instructions = """You are a game research assistant.

Your task is to answer user questions using the internal game database. You have access to two tools:

1. `retrieve_game`: use this to retrieve relevant game information from the internal database.
2. `evaluate_retrieval`: use this to judge whether the retrieved information is sufficient to answer the user's question.

When given a user question:
- First, always use `retrieve_game` to get relevant results.
- Then, call `evaluate_retrieval`, passing the original question and the list of retrieved documents. This will tell you whether the results are sufficient.
- Do not skip the evaluation step. Only after the evaluation, decide whether the system needs a fallback (e.g., to web search).

Wait for both tools to finish before forming a conclusion.

Return the tool results directly without answering the user’s question.
"""
                
        web_instructions = """You are a game industry research assistant. 
                        Use the game_web_search tool to find current information about games from 
                        the web when internal database information is insufficient. 
                        Provide comprehensive and accurate answers."""
        
        # First agent handles retrieval + eval
        self.rag_agent = ReActAgent(
            model_name=model_name,
            instructions=rag_instructions,
            tools=[retrieve_game, evaluate_retrieval],
            api_key=api_key,
            base_url=base_url,
            memory=self.memory
        )

        # Second agent handles final web fallback
        self.web_agent = ReActAgent(
            model_name=model_name,
            instructions=web_instructions,
            tools=[game_web_search],
            api_key=api_key,
            base_url=base_url,
            memory=self.memory
        )
        
        self.logger = logging.getLogger("GameAnswerAgent")

        # Prevent log propagation to root logger
        self.logger.propagate = False

        # Clear existing handlers **only on this logger**
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add your handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Set log level
        self.logger.setLevel(logging.DEBUG if log_level == "debug" else logging.INFO)
        self.logger.debug(f"Handlers on logger: {len(self.logger.handlers)}")
        
        self.logger.info(f"GameAnswerAgent initialized with model {model_name} and session ID {self.session_id}")
        self.logger.info(f"Debug Level: {log_level} - Handlers: {len(self.logger.handlers)}")
        
        # Create the main workflow state machine
        self.workflow = self._create_workflow()


    def _rag_step(self, state: GameAgentState) -> GameAgentState:
        """Step 1: Use RAG agent to retrieve and evaluate documents"""
        self.logger.debug(f"RAG Step: Processing query '{state['user_query']}'")
        
        # Run the RAG agent which will retrieve docs and evaluate them
        rag_run = self.rag_agent.invoke(state["user_query"], self.session_id)
        final_rag_state = rag_run.get_final_state()
        
        # Extract results from the RAG agent's final message
        rag_messages = final_rag_state.get("messages", [])
        
        # Find the last AI message with tool results
        rag_results = []
        evaluation_result = {"useful": False, "description": "No evaluation performed"}
        
        for message in reversed(rag_messages):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Look for tool messages that contain results
                tool_message_idx = len(rag_messages) - 1
                for i, msg in enumerate(rag_messages):
                    if hasattr(msg, 'tool_call_id'):
                        try:
                            result_data = json.loads(msg.content)
                            if isinstance(result_data, list):
                                rag_results = result_data
                            elif isinstance(result_data, dict) and 'useful' in result_data:
                                evaluation_result = result_data
                        except (json.JSONDecodeError, AttributeError):
                            pass
                break
        
        self.logger.debug(f"RAG results: {rag_results}")
        self.logger.debug(f"Evaluation result: {evaluation_result}")
        
        return {
            **state,
            "rag_results": rag_results,
            "evaluation_result": evaluation_result,
            "needs_web_search": not evaluation_result.get("useful", False)
        }
    
    def _web_step(self, state: GameAgentState) -> GameAgentState:
        """Step 2: Use Web agent to search for additional information"""
        self.logger.debug(f"Web Step: Searching web for '{state['user_query']}'")
        
        # Run the web agent
        web_run = self.web_agent.invoke(state["user_query"], session_id=self.session_id)
        final_web_state = web_run.get_final_state()
        
        # Extract web results
        web_messages = final_web_state.get("messages", [])
        web_results = {}
        
        for message in reversed(web_messages):
            if hasattr(message, 'content') and message.content:
                try:
                    # Try to parse as JSON first (tool result)
                    result_data = json.loads(message.content)
                    if isinstance(result_data, dict) and 'answer' in result_data:
                        web_results = result_data
                        break
                except json.JSONDecodeError:
                    # If not JSON, treat as final answer
                    web_results = {"answer": message.content}
                    break
        
        self.logger.debug(f"Web search results: {web_results}")
        
        return {
            **state,
            "web_results": web_results
        }
    
    def _answer_step(self, state: GameAgentState) -> GameAgentState:
        """Step 3: Generate final answer based on available information"""
        self.logger.debug("Answer Step: Generating final response")
        
        # Combine information from RAG and web search to create final answer
        final_answer = ""
        
        if state["evaluation_result"].get("useful", False):
            # Use RAG results
            rag_info = ""
            for doc in state["rag_results"]:
                rag_info += f"- {doc.get('Name', 'Unknown')} ({doc.get('Platform', 'Unknown')}, {doc.get('YearOfRelease', 'Unknown')}): {doc.get('Description', 'No description')}\n"
            
            final_answer = f"Based on our game database:\n{rag_info}"
        
        if state["needs_web_search"] and state.get("web_results"):
            # Use web results
            web_answer = state["web_results"].get("answer", "")
            if web_answer:
                if final_answer:
                    final_answer += f"\n\nAdditional information from web search:\n{web_answer}"
                else:
                    final_answer = f"Based on web search:\n{web_answer}"
        
        if not final_answer:
            final_answer = "I'm sorry, I couldn't find sufficient information to answer your question about games."
        
        self.logger.info(f"Final answer: {final_answer}")
        
        return {
            **state,
            "final_answer": final_answer
        }
    
    def _create_workflow(self) -> StateMachine[GameAgentState]:
        """Create the state machine workflow"""        
        machine: StateMachine[GameAgentState] = StateMachine(GameAgentState, logger=self.logger)

        # Create steps
        entry = EntryPoint[GameAgentState]()
        rag_step = Step[GameAgentState]("rag_step", self._rag_step)
        web_step = Step[GameAgentState]("web_step", self._web_step)
        answer_step = Step[GameAgentState]("answer_step", self._answer_step)
        termination = Termination[GameAgentState]()
        
        machine.add_steps([entry, rag_step, web_step, answer_step, termination])
        
        # Add transitions
        machine.connect(entry, rag_step)
        
        # After RAG, decide whether to use web search
        def check_evaluation(state: GameAgentState) -> Union[Step[GameAgentState], str]:
            if state.get("needs_web_search", False):
                return web_step  # Go to web search if RAG results are insufficient
            return answer_step  # Go directly to answer if RAG results are sufficient
        
        machine.connect(rag_step, [web_step, answer_step], check_evaluation)
        machine.connect(web_step, answer_step)  # After web search, generate answer
        machine.connect(answer_step, termination)  # End after generating answer
        
        return machine
    
    def ask(self, query: str) -> str:
        """
        Ask the GameAnswerAgent a question
        
        Args:
            query: The user's question about games
            
        Returns:
            The final answer string
        """
        self.logger.info(f"Starting workflow for query: '{query}'")
        
        initial_state: GameAgentState = {
            "user_query": query,
            "rag_results": [],
            "evaluation_result": {},
            "web_results": {},
            "final_answer": "",
            "needs_web_search": False
        }
        
        # Run the workflow
        run = self.workflow.run(initial_state)
        final_state = run.get_final_state()
        
        return final_state["final_answer"]
    
    def get_memory(self, session_id: str = None) -> List[GameAgentState]:
        """        Get all memory objects for a given session   
        Args:
            session_id: Optional session ID to filter memory objects
        Returns:
            List of GameAgentState objects from memory
        """
        if session_id is None:
            session_id = self.session_id

        return self.memory.get_all_objects(session_id)

    def print_memory(self, session_id: str = None) -> None:
        """Print all messages from memory for a given session"""
        for run in self.get_memory(session_id):
            print(run.get_final_state()["messages"])
