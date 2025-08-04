import os
import json
import re
from typing import List, Dict, Literal, TypedDict
from smolagents import ToolCallingAgent, tool, OpenAIServerModel
from protocol import QuoteItem, ParserResult
from tools.tools import (
    get_all_inventory
)
from json_repair import repair_json

from dotenv import load_dotenv

load_dotenv()

# --- Tool Definitions ---
@tool
def get_all_inventory_tool(as_of_date: str) -> Dict[str, Dict]:
    """
    Returns the entire inventory as of the specified date with detailed information.
    Args:
        as_of_date (str): ISO-formatted cutoff date (YYYY-MM-DD).
    Returns:
        Dict[str, Dict]: Dictionary with item names as keys and details (stock, buy_unit_price, sell_unit_price, category) as values.
    """
    return get_all_inventory(as_of_date)

@tool
def evaluate_inventory_parsing_tool(suggested_items: List[Dict], inventory_list: List[str]) -> Dict:
    """
    Evaluates whether all suggested matches are valid based on inventory.
    This tool internally uses an LLM to perform semantic validation.

    Args:
        suggested_items (List[Dict]): List of suggested items with 'name' and 'quantity' fields.
        inventory_list (List[str]): List of valid inventory item names.
    
    Returns: { "status": "ok" | "error", "problems": [list of strings] }
    """
    from openai import OpenAI
    from json import dumps, loads

    EVALUATION_PROMPT = f"""
You are the EvaluationAgent.

You are given:
- An official list of inventory item names:  
{inventory_list}

- A list of quote parsing suggestions made by another agent:  
{json.dumps(suggested_items, indent=2)}

Each suggestion includes a proposed match to an inventory item and the requested quantity.

---

## YOUR TASK

For each suggested item, determine if the match is valid, based on the following rules:

1. A valid suggestion must:
   - Match an item name **exactly** from the inventory list (case-insensitive, spacing-insensitive), OR
   - Be **semantically aligned** with an inventory item (e.g. "legal paper" → "Letter-sized paper"), and clearly refer to the same type and use.

2. An invalid suggestion occurs when:
   - The matched item does not exist in the inventory.
   - The original term is **ambiguous** or could plausibly refer to multiple inventory entries.
   - The suggested match is only partially correct or invented (e.g. "balloons").

3. Use common sense: if a human would ask for clarification, mark the suggestion as invalid and explain why.

---

## OUTPUT FORMAT

Return a **single JSON object** in this format:

```json
{{
  "status": "ok" | "error",
  "problems": [
    {{
      "original": "heavy cardstock (white)",
      "matched": "Cardstock",
      "reason": "Ambiguous: could also match '250 gsm cardstock'; no way to disambiguate."
    }},
    {{
      "original": "balloons",
      "matched": "balloons",
      "reason": "Not found in inventory and not semantically related to any known item."
    }}
  ]
}}
```

1. Do not include valid matches in the problems list.
2. A match is valid if it exists in the inventory or is clearly semantically aligned with an inventory item.
3. Only add items to the problems list if they are ambiguous, invalid, or invented.
4. Set "status": "ok" if the problems list is empty.
5. Set "status": "error" if there is at least one invalid suggestion in the problems list.

"""
    # Call LLM (replace with your OpenAIClient logic or smolagents LLM class)
    # For now, assume you're using openai-python SDK
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EVALUATION_PROMPT}],
    )

    text = response.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.DOTALL)
    
    return json.loads(repair_json(match.group() if match else text))




SYSTEM_INSTRUCTIONS = """
You are QuoteParserAgent.

You must reliably extract structured quote requests from natural language, using tool calls to guide your decisions.

Your goal is to transform a user's informal quote request into a validated, structured JSON object based on real inventory.

---

## PROCESS OVERVIEW

You must:

1. **Fetch the official inventory**
   - Use `get_all_inventory_tool(quote_request_date)` to retrieve the full inventory with item details.
   - The tool returns a dictionary where each item name maps to {"stock": int, "buy_unit_price": float, "sell_unit_price": float, "category": str}.
   - Extract the item names (dictionary keys) for matching purposes.
   - You must not invent or assume inventory items beyond this list.

2. **Parse and match requested items**
   - Extract each requested item and its quantity from the quote.
   - Use fuzzy matching to associate each extracted term with a valid inventory name.
     - A fuzzy match is valid **only if**:
       - It has a minimum similarity score of **80% token sort ratio**, OR
       - The mapping is clearly semantically aligned in meaning (e.g. "legal paper" → "Letter-sized paper").
     - If multiple candidates are possible and none is clearly superior, treat the item as **ambiguous** and unmatched.

3. **Evaluate your matches**
   - Use `evaluate_inventory_parsing_tool` to validate your suggested items.
   - Pass your matched list and the inventory list.
   - If the result is `status: "ok"`, proceed to final output.
   - If the result is `status: "error"`, revise your suggestions and retry (step 2.) — up to a maximum of 5 attempts.

4. **Ensure strict bucket assignment**
   - Each item must go into exactly **one** of the following buckets:
     - `"items"` → Confidently matched to a valid inventory item
     - `"unmatched_items"` → Either unmatched or ambiguous
   - Never include the same concept in both buckets.  
     For example: if "heavy cardstock" is matched to "Cardstock", it must not appear in `"unmatched_items"`.

5. **Avoid redundant calls**
   - Never call the same tool more than once in a single reasoning step.
   - Do not evaluate multiple item sets at once.
   - Call tools sequentially and reflect on each result before continuing.

---

## AT EACH STEP

- Clearly reason about:
  - What items were requested.
  - What matches you propose and why.
  - Which tool you will call next and for what purpose.
  - How to improve if evaluation fails.

---

## FINAL OUTPUT (strict format)

Once evaluation passes, return exactly one JSON object inside a code block:

Example:
```json
{
  "items": [
    { "name": "Inventory item name", "quantity": 100, "sell_unit_price": 5.0, "category": "paper" },
  ],
  "delivery_date": "YYYY-MM-DD",
  "unmatched_items": ["original term 1", "original term 2"],
  "status": "success" | "partial" | "declined"
}

"""

# SYSTEM_INSTRUCTIONS defined as above
# Tools: get_all_inventory_tool and evaluate_inventory_parsing_tool are registered

class QuoteRequestParserAgent:
    def __init__(self, verbosity_level: int = 0):
        self.model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

        self.agent = ToolCallingAgent(
            tools=[
                get_all_inventory_tool,
                evaluate_inventory_parsing_tool
            ],
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            max_steps=20,
            verbosity_level=verbosity_level,
            #max_tool_threads=1,
        )

    def run(self, quote_request: str, quote_request_date: str = "2025-08-01") -> ParserResult:
        try:
            result = self.agent.run(
                task=f"Parse and evaluate quote request: '{quote_request}'",
                additional_args={
                    "quote_request": quote_request,
                    "quote_request_date": quote_request_date
                }
            )

            # Parse the JSON result from the agent
            parsed_data = self._parse_agent_response(result)
            
            # Convert to QuoteItem objects
            quote_items = []
            for item_data in parsed_data.get("items", []):
                quote_items.append(QuoteItem(
                    name=item_data.get("name", ""),
                    quantity=item_data.get("quantity", 0), 
                    unit_price=item_data.get("sell_unit_price", 0.0),
                    category=item_data.get("category", None)                   
                ))
            
            # Return structured ParserResult
            return ParserResult(
                items=quote_items,                
                delivery_date=parsed_data.get("delivery_date", None),
                unmatched_items=parsed_data.get("unmatched_items", []),
                status=parsed_data.get("status", "success" if quote_items else "declined")
            )

        except Exception as e:
            print(f"Error in QuoteRequestParserAgent: {e}")
            return ParserResult(
                items=[],
                delivery_date="01-01-1970",  # Default date if parsing fails
                unmatched_items=[],
                status="declined",
                error_message=str(e)
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
            print(f"Failed to parse agent response as JSON: {e}")
            
        # Fallback: return empty structure
        return {
            "items": [],
            "delivery_date": None,
            "unmatched_items": [],
            "status": "declined"
        }
