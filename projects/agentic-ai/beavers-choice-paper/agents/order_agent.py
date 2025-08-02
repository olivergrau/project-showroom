import os
import json
import re
from typing import List, Dict, Literal, TypedDict
from smolagents import CodeAgent, tool, OpenAIServerModel
from protocol import QuoteItem, QuoteResult, OrderResult
from tools.tools import (
    create_transaction
)
from json_repair import repair_json

from dotenv import load_dotenv

load_dotenv()

# --- Tool Definitions ---

class TransactionData(TypedDict):
    item_name: str
    transaction_type: Literal["stock_orders", "sales"]
    quantity: int
    price: float
    date: str  # ISO format (YYYY-MM-DD)

@tool
def create_transaction_tool(data: List[TransactionData]) -> List[int]:
    """
    Records one or more transactions of type 'stock_orders' or 'sales' into the database.

    Args:
        data (List[TransactionData]): A list of transaction records to create. Each must contain:
            - item_name (str): Name of the item involved
            - transaction_type (str): 'stock_orders' or 'sales'
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
            transaction_type=record["transaction_type"],
            quantity=record["quantity"],
            price=record["price"],
            date=record["date"],
        )
        inserted_ids.append(inserted_id)
    return inserted_ids

# --- Prompt for OrderAgent ---

SYSTEM_INSTRUCTIONS = """
You are OrderAgent. Your job is to finalize sales by creating transaction records for accepted quotes.

You have access to these tools:
- create_transaction_tool(data): creates sales transactions; data must be a list of transaction records

You will receive:
- quote_result: dict containing total_price, currency, line_items, and notes
- quote_request_date: string in ISO format YYYY-MM-DD (the date when the quote was requested - use this as "today")

Your task:

1. Extract line items from the quote_result:
   a. Each line item contains name, quantity, and pricing information (unit_price, subtotal)
   b. Use the subtotal if available, otherwise calculate price = unit_price * quantity

2. Create sales transactions:
   a. For each line item, create a transaction record with:
      - item_name: the item name
      - transaction_type: "sales"
      - quantity: the quantity sold
      - price: the subtotal (or calculated price) for this line item
      - date: the quote_request_date (not today's date)
   b. Call create_transaction_tool with the list of transaction records

3. Generate order summary:
   a. Create an order_id using format "ORD-{timestamp}" (use current timestamp)
   b. Create a summary message listing all transactions created with their IDs
   c. Include total amount and number of items

4. Create a Python dictionary with these fields:
   - success: True if all transactions were created successfully
   - order_id: generated order identifier
   - message: summary of transactions created, including transaction IDs

Assign this dictionary to a variable called `final_result`, and print it using Python code.

Example:
```python
final_result = {
    "success": True,
    "order_id": "ORD-1722441600",
    "message": "Order completed successfully. Created 2 transactions: [134, 135]. Total: $30.00 for 3 items."
}
print(final_result)
```
"""

class OrderAgent:
    def __init__(self):
        self.model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        self.agent = CodeAgent(
            tools=[create_transaction_tool],
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            max_steps=20,
            verbosity_level=0,
            additional_authorized_imports=["json", "os", "typing", "datetime", "pandas", "time"]
        )

    def run(self, quote_result: QuoteResult, quote_request_date: str = "2025-08-01") -> OrderResult:
        """
        Finalize the order by creating sales transactions for the accepted quote.
        
        Args:
            quote_result: QuoteResult object containing the accepted quote details
            quote_request_date: Date when the quote was requested in ISO format (YYYY-MM-DD)
            
        Returns:
            OrderResult: Structured result following message protocol
        """
        try:
            quote_data = quote_result.dict()
            inputs = {
                "quote_result": quote_data,
                "quote_request_date": quote_request_date
            }

            result = self.agent.run(
                task="Create sales transactions for the accepted quote and generate order summary.",
                additional_args=inputs
            )

            # Parse the agent's JSON response
            parsed_data = self._parse_agent_response(result)
            
            # Return structured OrderResult
            return OrderResult(
                success=parsed_data.get("success", False),
                order_id=parsed_data.get("order_id", None),
                message=parsed_data.get("message", "Order processing completed.")
            )
            
        except Exception as e:
            print(f"Error in order agent: {e}")
            # Return failed order on error
            return OrderResult(
                success=False,
                order_id=None,
                message=f"Order failed: {str(e)}"
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
            print(f"Failed to parse order agent response as JSON: {e}")
            
        # Fallback: return failure structure
        return {
            "success": False,
            "order_id": None,
            "message": "Failed to parse order response."
        }
