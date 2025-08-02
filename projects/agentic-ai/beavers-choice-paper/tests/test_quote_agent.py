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

from agents.quote_agent import QuoteAgent
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
    QuoteItem(name="Standard copy paper", quantity=150),    # Should trigger bulk discount (>=100)
    QuoteItem(name="Glossy paper", quantity=25),           # Regular pricing
    QuoteItem(name="Cardstock", quantity=500),             # Should trigger larger bulk discount (>=500)
    QuoteItem(name="A4 paper", quantity=75),               # Regular pricing (below bulk threshold)
    QuoteItem(name="Paper plates", quantity=200),          # Bulk discount for product category
]

print("Testing QuoteAgent with various quantity levels...\n")
print("Test Items:")
for item in test_items:
    print(f"  - {item.name}: {item.quantity} units")
print()

# --- Run the agent ---
print("Running QuoteAgent test...\n")

# Create fresh quote agent instance
quote_agent = QuoteAgent(verbosity_level=2)

result = quote_agent.run(test_items)

# --- Pretty print result ---
import json
print("QuoteAgent Output:")
print(result.model_dump_json(indent=2))

# --- Additional analysis ---
print("\n" + "="*50)
print("QUOTE ANALYSIS")
print("="*50)
print(f"💰 Total Quote Amount: ${result.total_price:.2f}")
print(f"💱 Currency: {result.currency}")
print(f"📋 Number of Line Items: {len(result.line_items)}")
print(f"📝 Notes: {result.notes}")

if result.line_items:
    print("\n📊 Line Item Breakdown:")
    for item in result.line_items:
        #unit_text = f" ({item.unit})" if item.unit else ""
        price_text = f" @ ${item.unit_price:.2f}" if item.unit_price else ""
        discount_text = f" ({item.discount_percent:.1f}% discount)" if item.discount_percent and item.discount_percent > 0 else ""
        subtotal_text = f" = ${item.subtotal:.2f}" if item.subtotal else ""
        
        print(f"  • {item.name}: {item.quantity} {price_text}{discount_text}{subtotal_text}")        

print("\n✅ QuoteAgent test completed!")
