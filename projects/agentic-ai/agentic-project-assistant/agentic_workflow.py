# agentic_workflow.py

# Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
    Route,
)
from agents.openai_service import OpenAIService
from utils.logging_config import logger
from config import load_openai_api_key, load_openai_base_url
import os


# Load OpenAI credentials using the shared config helper
openai_api_key = load_openai_api_key()
openai_base_url = load_openai_base_url()

# Create a single OpenAIService instance to share across agents
openai_service = OpenAIService(api_key=openai_api_key, base_url=openai_base_url)

# load the product spec
# Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
product_spec_path = "Product-Spec-Email-Router.txt"
if not os.path.exists(product_spec_path):
    raise FileNotFoundError(f"Product spec file not found at {product_spec_path}")

with open(product_spec_path, "r") as file:
    product_spec = file.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    knowledge=knowledge_action_planning, openai_service=openai_service
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"Here is the product spec: {product_spec}"
)
# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
    openai_service=openai_service,
)

# Product Manager - Evaluation Agent
# Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
evaluation_criteria_product_manager = (
    "The answer should be user stories that follow this structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]. "
    "Each user story should be clear, concise, and focused on a specific user need. "
    "The user stories should be relevant to the product spec provided."
)
product_manager_evaluation_agent = EvaluationAgent(
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    openai_service=openai_service,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
    openai_service=openai_service,
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)

# Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
program_manager_evaluation_agent = EvaluationAgent(
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The answer should be product features that follow the following structure: "
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user"
    ),
    worker_agent=program_manager_knowledge_agent,
    openai_service=openai_service,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
    openai_service=openai_service,
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
# Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
development_engineer_evaluation_agent = EvaluationAgent(
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer should be tasks following this exact structure: "
        "Task ID: A unique identifier for tracking purposes\n"
        "Task Title: Brief description of the specific development work\n"
        "Related User Story: Reference to the parent user story\n"
        "Description: Detailed explanation of the technical work required\n"
        "Acceptance Criteria: Specific requirements that must be met for completion\n"
        "Estimated Effort: Time or complexity estimation\n"
        "Dependencies: Any tasks that must be completed first"
    ),
    worker_agent=development_engineer_knowledge_agent,
    openai_service=openai_service,
)


# Job function persona support functions
# Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)
    evaluation = product_manager_evaluation_agent.iterate(response)
    return evaluation


def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)
    evaluation = program_manager_evaluation_agent.iterate(response)
    return evaluation


def development_engineer_support_function(query):
    response = development_engineer_knowledge_agent.respond(query)
    evaluation = development_engineer_evaluation_agent.iterate(response)
    return evaluation


# Define the routes and instantiate the routing agent
routes = [
    Route(
        name="Product Manager",
        description="Responsible for defining product personas and user stories only.",
        func=product_manager_support_function,
    ),
    Route(
        name="Program Manager",
        description="A Program Manager, who is responsible for defining the features for a product.",
        func=program_manager_support_function,
    ),
    Route(
        name="Development Engineer",
        description="A Development Engineer, who is responsible for defining the development tasks for a product.",
        func=development_engineer_support_function,
    ),
]

routing_agent = RoutingAgent(agents=routes, openai_service=openai_service)



# Run the workflow

logger.info("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
logger.info("Task to complete in this workflow, workflow prompt = %s", workflow_prompt)

logger.info("\nDefining workflow steps from the workflow prompt")
# Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
extracted_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

# print out completed steps
logger.info("Extracted steps from the workflow prompt: %s\n", extracted_steps)
logger.info("Executing each step in the workflow...")

# Start with an empty context
context_prompt = ""

# Loop through the extracted steps
for i, step in enumerate(extracted_steps):
    # Combine previous context with the current step
    full_prompt = context_prompt + f"\n{step}".strip()
    logger.debug("Full prompt for step %d: %s", i + 1, full_prompt)
    logger.info("Executing step %d: %s", i + 1, step)
    result = routing_agent.route(full_prompt)

    # Store the result
    completed_steps.append(result)

    # Update context with the final response from this step
    if isinstance(result, dict) and "final_response" in result:
        context_prompt += "\n" + result["final_response"]
    else:
        # Optional safety check
        context_prompt += "\n" + str(result)

    logger.info("Result of step '%s': %s", step, result)

logger.info("Workflow completed successfully.")

# Print complete workflow results
logger.info("\n*** Workflow Results ***")

for i, step in enumerate(completed_steps, start=1):
    logger.info("Step %d: %s", i, step)

logger.info("Final output of the workflow: %s", completed_steps[-1])
logger.info("\n*** Workflow execution finished ***\n")
