import os
import json
import re
from typing import List, Dict, Literal, TypedDict
from smolagents import CodeAgent, tool, OpenAIServerModel
from protocol import QuoteItem, InventoryCheckResult
from tools.tools import (
    get_stock_level,
    get_cash_balance,
    get_supplier_delivery_date,
    create_transaction,
    get_buy_unit_price
)
from json_repair import repair_json

from dotenv import load_dotenv

load_dotenv()

# --- Tool Definitions ---
@tool
def get_stock_level_tool(item_name: str, as_of_date: str) -> Dict[str, int]:
    """
    Returns the current stock level for a given item as of the specified date.

    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): ISO-formatted cutoff date (YYYY-MM-DD).

    Returns:
        int: Current stock level for the item, e.g. 17
    """
    df = get_stock_level(item_name, as_of_date)
    stock = int(df.iloc[0]["current_stock"]) if not df.empty else 0
    return stock

@tool
def get_cash_balance_tool(as_of_date: str) -> float:
    """
    Returns the net cash balance as of the given date.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD).

    Returns:
        float: Net cash available at the given date.
    """
    return get_cash_balance(as_of_date)

@tool
def get_supplier_delivery_date_tool(requested_date: str, quantity: int) -> str:
    """
    Returns estimated supplier delivery date based on requested start date and quantity.

    Args:
        requested_date (str): The starting date (usually today) in ISO format (YYYY-MM-DD).
        quantity (int): Number of units to order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    return get_supplier_delivery_date(requested_date, quantity)

@tool
def get_buy_unit_price_tool(item_name: str) -> float:
    """
    Returns the buy unit price for a given item from the inventory.

    Args:
        item_name (str): The name of the item to get buy pricing for.

    Returns:
        float: Buy unit price for the item, or 0.0 if not found.
    """
    df = get_buy_unit_price(item_name)
    if not df.empty:
        return float(df.iloc[0]["buy_unit_price"])
    else:
        return 0.0


class TransactionData(TypedDict):
    item_name: str
    transaction_type: Literal["stock_orders", "sales"]
    quantity: int
    price: float
    date: str  # ISO format (YYYY-MM-DD)

@tool
def create_transaction_tool(data: List[TransactionData]) -> List[int]:
    """
    Records one or more transactions of type 'stock_orders' only into the database.

    Args:
        data (List[TransactionData]): A list of transaction records to create. Each must contain:
            - item_name (str): Name of the item involved
            - transaction_type (str): 'stock_orders'
            - quantity (int): Number of units involved
            - price (float): Total price for the transaction
            - date (str): Transaction date in ISO format (YYYY-MM-DD)

    Returns:
        List[int]: A list of IDs for the newly inserted transactions.
    """
    inserted_ids = []
    for record in data:
        inserted_id = create_transaction(
            item_name=record["item_name"],
            transaction_type="stock_orders",  # Only stock orders
            quantity=record["quantity"],
            price=record["price"],
            date=record["date"],
        )
        inserted_ids.append(inserted_id)
    return inserted_ids



# --- Prompt for InventoryAgent ---

SYSTEM_INSTRUCTIONS = """
You are InventoryAgent. Your job is to determine which requested items can be fulfilled based on stock availability, supplier delivery timing, and cash reserves.

You have access to these tools:
- get_stock_level_tool(item_name, as_of_date): returns available stock (as int) for a specific item as of the given date
- get_supplier_delivery_date_tool(requested_date, quantity): returns delivery date in ISO format (YYYY-MM-DD) as a string
- get_cash_balance_tool(as_of_date): returns current cash balance as a float
- get_buy_unit_price_tool(item_name): returns the buy unit price (float) for an item
- create_transaction_tool(data): creates stock order transactions; data must be a list of {"item_name": ..., "quantity": ...}

You will receive:
- quote_items: list of dicts, each with 'name' and 'quantity'
- requested_delivery_date: string in ISO format YYYY-MM-DD
- quote_request_date: string in ISO format YYYY-MM-DD (the date when the quote was requested - use this as "today")

Your task:

1. For each item in quote_items:
   a. Use get_stock_level_tool to check current stock level as of quote_request_date.
   b. If sufficient stock is available → mark as fulfillable.
   c. If not in stock:
    - Call get_supplier_delivery_date_tool(requested_date, quantity) to simulate delivery date from supplier.
        - `requested_date` is the quote_request_date (not the delivery date).
        - `quantity` is the amount needed for the item that is not in stock.
        - This tool does not require the item name.
        - If the estimated delivery date is **before** the customer's requested delivery date, the item is restockable.
        - Otherwise, mark the item as unfulfillable.
        - If the supplier's delivery date is **before or on** the `requested_delivery_date`, this item is **restockable in time**.
        - However, it is **NOT fulfillable yet** – unless cash is available and a restocking transaction is issued (see next step).

2. If any items are not in stock and they are restockable (found out in 1.c):
   a. Use get_cash_balance_tool(as_of_date=quote_request_date) to get current cash.
   b. If cash is sufficient to restock all required items:
      Create stock order transactions:
        a. For each line item that needs restocking, create a transaction record with:
            - item_name: the item name
            - transaction_type: "stock_orders"
            - quantity: the quantity needed for restocking
            - price: quantity * get_buy_unit_price_tool(item_name) (use buy price for stock orders)
            - date: the quote_request_date
        b. Call create_transaction_tool with the list of transaction records to create stock orders.
        c. Save the transaction details for the final result
        d. Mark these items as fulfillable since they will be restocked
   c. If cash is insufficient, mark restockable items as unfulfillable.

   DO NOT FORGET: to execute the created stock order transactions using create_transaction_tool.

3. If all items are already in stock, **do not** call get_cash_balance_tool or create_transaction_tool.

4. At the end, create a Python dictionary with the following fields:
   - fulfillable_items: list of item dicts (same format as input)
   - unfulfillable_items: list of item dicts
   - all_items_fulfillable: true if all requested items are fulfillable
   - some_items_fulfillable: true if at least one but not all items are fulfillable
   - no_items_fulfillable: true if none are fulfillable
   - restockable_items: list of item names that are restockable but not yet ordered
   - stock_orders: list of transaction dicts that were created (if any)

Assign this dictionary to a variable called `final_result`, and print it using Python code.

Example:
```python
final_result = {
    "fulfillable_items": [...],
    "unfulfillable_items": [...],
    "all_items_fulfillable": True,
    "some_items_fulfillable": False,
    "no_items_fulfillable": False,
    "restockable_items": ["item1", "item2"],
    "stock_orders": [
        {"item_name": "item3", "transaction_type": "stock_orders", "quantity": 10, "price": 50.0, "date": "2025-08-01"},
        {"item_name": "item4", "transaction_type": "stock_orders", "quantity": 5, "price": 25.0, "date": "2025-08-01"}
    ]
}
print(final_result)
"""

class InventoryAgent:
    def __init__(self, verbosity_level: int = 0):
        self.model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        self.agent = CodeAgent(
            tools=[get_cash_balance_tool, get_stock_level_tool, get_supplier_delivery_date_tool, get_buy_unit_price_tool, create_transaction_tool],
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            max_steps=20,
            verbosity_level=verbosity_level,
            additional_authorized_imports=["json", "os", "typing", "datetime", "pandas"]
        )

    def run(self, quote_items: List[QuoteItem], delivery_date: str = "2025-08-01", quote_request_date: str = "2025-08-01") -> InventoryCheckResult:
        """
        Process quote items and determine fulfillment feasibility.
        Args:
            quote_items: List of QuoteItem objects to check
            delivery_date: Requested delivery date in ISO format (YYYY-MM-DD)
            quote_request_date: Date when the quote was requested in ISO format (YYYY-MM-DD)
        Returns:
            InventoryCheckResult: Structured result following message protocol
        """
        try:
            items = [item.model_dump() for item in quote_items]
            inputs = {
                "quote_items": items,
                "requested_delivery_date": delivery_date,
                "quote_request_date": quote_request_date
            }

            result = self.agent.run(
                task="Determine which items can be fulfilled based on stock and delivery date.",
                additional_args=inputs
            )

            # Parse the agent's JSON response
            parsed_data = self._parse_agent_response(result)
            
            # Convert back to QuoteItem objects
            fulfillable_items = []
            unfulfillable_items = []
            
            # Create lookup for original items
            item_lookup = {item.name: item for item in quote_items}
            
            for item_data in parsed_data.get("fulfillable_items", []):
                item_name = item_data.get("name", "")
                if item_name in item_lookup:
                    fulfillable_items.append(item_lookup[item_name])
            
            for item_data in parsed_data.get("unfulfillable_items", []):
                item_name = item_data.get("name", "")
                if item_name in item_lookup:
                    unfulfillable_items.append(item_lookup[item_name])
            
            # Collect restockable items and stock orders
            restockable_items = parsed_data.get("restockable_items", [])
            stock_orders = parsed_data.get("stock_orders", [])
            
            # Return structured InventoryCheckResult
            return InventoryCheckResult(
                fulfillable_items=fulfillable_items,
                unfulfillable_items=unfulfillable_items,
                restockable_items=restockable_items,
                stock_orders=stock_orders
            )
            
        except Exception as e:
            print(f"Error in inventory agent: {e}")
            # Return all items as unfulfillable on error
            return InventoryCheckResult(
                fulfillable_items=[],
                unfulfillable_items=quote_items,
                restockable_items=[],
                stock_orders=[]
            )
    
    def _parse_agent_response(self, response: str) -> Dict:
        """
        Parse the agent's JSON response and extract structured data.
        """
        try:
            # Try to extract JSON from the response
            if isinstance(response, dict):
                return response
            
            # If response is a string, try to parse JSON
            if isinstance(response, str):
                # Look for JSON-like content
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(repair_json(json_match.group()))
                
                # Fallback: try parsing the entire response as JSON
                return json.loads(repair_json(response))
                
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Failed to parse inventory agent response as JSON: {e}")
            
        # Fallback: return empty structure
        return {
            "fulfillable_items": [],
            "unfulfillable_items": [],
            "restockable_items": [],
            "stock_orders": []
        }
