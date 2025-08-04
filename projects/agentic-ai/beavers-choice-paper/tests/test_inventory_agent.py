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

from agents.inventory_agent import InventoryAgent
from protocol import QuoteItem
from dotenv import load_dotenv

load_dotenv()

from tools.tools import db_engine, init_database

# Initialize the database engine and create sample inventory
print("Initializing database and generating sample inventory...\n")
init_database(db_engine=db_engine, seed=0, debug=True)

print("Sample inventory generated successfully.\n")

# --- Simulate a user quote request ---
test_items = [
    QuoteItem(name="Standard copy paper", quantity=1000, unit_price=0.04),
    QuoteItem(name="Glossy paper", quantity=5, unit_price=0.20),
    QuoteItem(name="Notepads", quantity=8, unit_price=0.50),
    QuoteItem(name="Letterhead paper", quantity=3, unit_price=0.10),
]

# Requested delivery date — adjust if needed
requested_delivery_date = "2025-08-15"
quote_request_date = "2025-08-01"  # Date when the quote was requested

print(f"Requested delivery date: {requested_delivery_date}")
print(f"Quote request date: {quote_request_date}\n")

# --- Run the agent ---
print("Running InventoryAgent test...\n")

# Create fresh inventory agent instance
inventory_agent = InventoryAgent(verbosity_level=2)

result = inventory_agent.run(
    test_items, delivery_date=requested_delivery_date, quote_request_date=quote_request_date)

# --- Pretty print result ---
import json
print("InventoryAgent Output:")
print(result.model_dump_json(indent=2))
