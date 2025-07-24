import os
import sys
import argparse
import logging
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CHROMA_OPENAI_API_KEY = os.getenv("CHROMA_OPENAI_API_KEY")
CHROMA_OPENAI_API_BASE = os.getenv("CHROMA_OPENAI_API_BASE")

# Import project modules
from framework.react_agent import ReActAgent
from framework.llm import LLM
from framework.messages import UserMessage, SystemMessage, ToolMessage, AIMessage
from framework.tooling import tool
from tavily import TavilyClient
from datetime import datetime
from typing import Dict
import chromadb
from chromadb.utils import embedding_functions
from framework.evaluation import AgentEvaluator, TestCase
from json_repair import repair_json
from framework.game_answer_agent import GameAnswerAgent

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="chromadb")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=CHROMA_OPENAI_API_KEY,
    model_name="text-embedding-3-small",
    api_base=CHROMA_OPENAI_API_BASE or "https://api.openai.com/v1"
)
collection = chroma_client.get_collection("udaplay")
client = TavilyClient(api_key=TAVILY_API_KEY)

# --- Tool Definitions ---
@tool(
    name="retrieve_game",
    description="Semantic search: Finds most results in the vector DB"
)
def retrieve_game(query: str):
    results = collection.query(query_texts=query, n_results=1)
    formatted_results = []
    for metadata in results["metadatas"][0]:
        formatted_results.append({
            "Name": metadata.get("Name"),
            "Platform": metadata.get("Platform"),
            "YearOfRelease": metadata.get("YearOfRelease"),
            "Description": metadata.get("Description")
        })
    return formatted_results

@tool(
    name="evaluate_retrieval",
    description="Based on the user's question and on the list of retrieved documents, it will analyze the usability of the documents to respond to that question."
)
def evaluate_retrieval(question: str, retrieved_docs: str):
    evaluator = AgentEvaluator(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    try:
        parsed_docs = json.loads(repair_json(retrieved_docs))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON passed to evaluate_retrieval: {e}. Original input: {retrieved_docs} ")
    docs_text = ""
    if parsed_docs:
        for i, doc in enumerate(parsed_docs, start=1):
            if isinstance(doc, dict):
                platform = doc.get('Platform', 'Unknown')
                name = doc.get('Name', 'Unknown')
                year = doc.get('YearOfRelease', 'Unknown')
                description = doc.get('Description', 'No description')
                docs_text += f"Document {i}:\n- Platform: {platform}\n- Name: {name}\n- Year: {year}\n- Description: {description}\n\n"
            else:
                docs_text += f"Document {i}: {str(doc)}\n\n"
    else:
        docs_text = "No documents retrieved."
    test_case = TestCase(
        id="doc_eval",
        description="Evaluate if retrieved documents are sufficient to answer the user's question",
        user_query=question,
        expected_tools=["retrieve_game"],
        reference_answer="Documents should contain relevant information to answer the question"
    )
    evaluation_result = evaluator.evaluate_final_response(
        test_case=test_case,
        agent_response=docs_text,
        execution_time=0.0,
        total_tokens=0,
    )
    return {
        "useful": evaluation_result.task_completion.task_completed,
        "description": evaluation_result.feedback
    }

@tool
def game_web_search(query: str, search_depth: str = "advanced") -> Dict:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    search_result = client.search(
        query=query,
        search_depth=search_depth,
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
    formatted_results = {
        "answer": search_result.get("answer", ""),
        "results": search_result.get("results", []),
        "search_metadata": {
            "timestamp": datetime.now().isoformat(),
            "query": query
        }
    }
    return formatted_results

# --- Main CLI Entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="Ask Udaplay GameAnswerAgent a question.")
    parser.add_argument("question", type=str, help="Your question about video games")
    parser.add_argument("--log-level", type=str, choices=["debug", "info"], default="info", help="Logging verbosity")
    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger("UdaplayCLI")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.log_level == "debug" else logging.INFO)

    logger.info("Initializing GameAnswerAgent...")
    game_agent = GameAnswerAgent(
        model_name="gpt-4o-mini",
        retrieve_game=retrieve_game,
        evaluate_retrieval=evaluate_retrieval,
        game_web_search=game_web_search,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        log_level=args.log_level
    )

    logger.info(f"Question: {args.question}")
    answer = game_agent.ask(args.question)
    logger.info(f"Final answer: {answer}")

if __name__ == "__main__":
    main()