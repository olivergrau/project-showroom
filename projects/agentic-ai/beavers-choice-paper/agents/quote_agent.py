import os
import json
import re
from typing import List, Dict, Literal, TypedDict
from smolagents import CodeAgent, tool, OpenAIServerModel
from protocol import QuoteItem, QuoteResult
from tools.tools import (
    search_quote_history,
    get_unit_price
)
from json_repair import repair_json

from dotenv import load_dotenv

load_dotenv()

# --- Tool Definitions ---

@tool
def get_unit_price_tool(item_name: str) -> Dict[str, float]:
    """
    Returns the unit price for a given item from the inventory.

    Args:
        item_name (str): The name of the item to get pricing for.

    Returns:
        Dict[str, float]: Dictionary with 'unit_price' field, or 0.0 if not found.
    """
    df = get_unit_price(item_name)
    if not df.empty:
        return {"unit_price": float(df.iloc[0]["unit_price"])}
    else:
        return {"unit_price": 0.0}

@tool
def search_quote_history_tool(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Search historical quotes for similar items or patterns to inform pricing decisions.

    Args:
        search_terms (List[str]): List of terms to search for in historical quotes.
        limit (int): Maximum number of historical quotes to return.

    Returns:
        List[Dict]: List of historical quote records with pricing and discount information.
    """
    return search_quote_history(search_terms, limit)

# --- Prompt for QuoteAgent ---

SYSTEM_INSTRUCTIONS = """
You are QuoteAgent. Your job is to calculate quotes for fulfillable items based on unit prices and historical pricing patterns.

You have access to these tools:
- get_unit_price_tool(item_name): returns unit price as {"unit_price": float_value} for a specific item
- search_quote_history_tool(search_terms, limit): returns historical quotes matching search terms

You will receive:
- quote_items: list of dicts, each with 'name' and 'quantity' fields

Your task:

1. For each item in quote_items:
   a. Use get_unit_price_tool(item_name) to get the base unit price.
   b. Calculate base cost = unit_price * quantity.
   c. If unit_price is 0.0, set base cost to 0.0 and note "No price found for {item_name}".

2. Search for historical quotes:
   a. Extract item names from quote_items to use as search terms.
   b. Use search_quote_history_tool(search_terms=[item_names], limit=3) to find similar past quotes.
   c. Look for discount patterns in the quote_explanation field of historical quotes.

3. Apply bulk discounts based on historical patterns:
   a. If any historical quote mentions "bulk discount", "volume discount", or similar terms AND the current quantity is >=100 units, apply a 10% discount.
   b. If quantity is >=500 units and historical quotes show large order discounts, apply a 15% discount.
   c. If no historical discount patterns are found, only apply bulk discount for quantities >=100 (10% off).

4. Calculate the final quote:
   a. Sum up all line item costs (after discounts).
   b. Create detailed line items as a list of dicts with {"name", "quantity", "unit_price", "discount_percent", "subtotal"} fields matching QuoteItem structure.

5. Create a Python dictionary with these fields:
   - total_price: float (total amount for all items)
   - currency: "USD"
   - line_items: list of dicts with {"name", "quantity", "unit", "requested_by", "unit_price", "discount_percent", "subtotal"} fields (matching QuoteItem structure)
   - notes: string explaining any discounts applied or pricing decisions

Assign this dictionary to a variable called `final_result`, and print it using Python code.

Example:
```python
final_result = {
    "success": True,
    "total_price": 87.50,
    "currency": "USD",
    "line_items": [
        {"name": "A4 paper", "quantity": 100, "unit_price": 0.05, "discount_percent": 10.0, "subtotal": 4.50}
    ],
    "notes": "10% bulk discount applied for orders >= 100 units based on historical pricing patterns."
}
print(final_result)
```

Set success to True if all items have a non-zero unit price and the quote was generated successfully, otherwise set it to False.

"""

class QuoteAgent:
    def __init__(self, verbosity_level: int = 0):
        self.model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        self.agent = CodeAgent(
            tools=[get_unit_price_tool, search_quote_history_tool],
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            max_steps=20,
            verbosity_level=verbosity_level,
            additional_authorized_imports=["json", "os", "typing", "datetime", "pandas"]
        )

    def run(self, quote_items: List[QuoteItem]) -> QuoteResult:
        """
        Generate a quote for the given items with pricing and discount logic.
        
        Args:
            quote_items: List of QuoteItem objects to calculate pricing for
            
        Returns:
            QuoteResult: Structured result following message protocol
        """
        try:
            items = [item.model_dump() for item in quote_items]
            inputs = {
                "quote_items": items
            }

            result = self.agent.run(
                task="Calculate quote with pricing and bulk discounts based on historical patterns.",
                additional_args=inputs
            )

            # Parse the agent's JSON response
            parsed_data = self._parse_agent_response(result)
            
            # Convert line_items dictionaries to QuoteItem objects
            line_items = []
            for item_data in parsed_data.get("line_items", []):
                try:
                    line_items.append(QuoteItem(**item_data))
                except Exception as e:
                    print(f"Warning: Could not convert line item to QuoteItem: {e}")
                    # Skip invalid line items
                    continue
            
            # Return structured QuoteResult
            return QuoteResult(
                success=parsed_data.get("success", False),
                total_price=parsed_data.get("total_price", 0.0),
                currency=parsed_data.get("currency", "USD"),
                line_items=line_items,
                notes=parsed_data.get("notes", "Quote generated successfully.")
            )
            
        except Exception as e:
            print(f"Error in quote agent: {e}")
            # Return empty quote on error
            return QuoteResult(
                total_price=0.0,
                currency="USD",
                line_items=[],
                notes=f"Error generating quote: {str(e)}"
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
            print(f"Failed to parse quote agent response as JSON: {e}")
            
        # Fallback: return empty structure
        return {
            "total_price": 0.0,
            "currency": "USD",
            "line_items": [],
            "notes": "Failed to parse quote response."
        }
