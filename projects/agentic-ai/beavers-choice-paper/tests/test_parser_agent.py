# set cwd directory to the root of the project
import os
import sys
from pathlib import Path

# Get the absolute path to the current file (inside tests/)
current_file_path = Path(__file__).resolve()

# Set working directory to the parent of 'tests'
project_root = current_file_path.parent.parent
os.chdir(project_root)

# (Optional) Add root to sys.path if you want to import project modules directly
sys.path.insert(0, str(project_root))

print(f"Changed working directory to: {os.getcwd()}")

from agents.parser_agent import QuoteRequestParserAgent
from protocol import QuoteItem
from dotenv import load_dotenv

load_dotenv()

from tools.tools import db_engine, init_database

# Initialize the database engine and create sample inventory
print("Initializing database and generating sample inventory...\n")
init_database(db_engine=db_engine, seed=0, debug=True)

print("Sample inventory generated successfully.\n")

# Requested delivery date — adjust if needed
request_quote = """I would like to request the following paper supplies for the ceremony: 

- 200 sheets of A4 glossy paper
- 100 sheets of heavy cardstock (white)
- 100 balloons

I need these supplies delivered by August 15, 2025. Thank you."""

# --- Run the agent ---
print("Running QuoteRequestParserAgent test...\n")

quote_request_date = "2025-08-01"  # Example date, adjust as needed
print(f"Quote Request Date: {quote_request_date}\n")

# Create fresh parser agent instance
parser_agent = QuoteRequestParserAgent(verbosity_level=2)

# Call the parser agent with the quote request and date
result = parser_agent.run(
    quote_request=request_quote,
    quote_request_date=quote_request_date
)

# --- Pretty print result ---
import json
print("ParserAgent Output:")
print(result.model_dump_json(indent=2))
