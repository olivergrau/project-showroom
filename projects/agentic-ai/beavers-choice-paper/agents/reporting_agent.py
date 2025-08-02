"""
ReportingAgent for Beaver's Choice Paper Company
Generates financial reports and analytics using the smolagents framework.
"""

import json
import os
from typing import Any, Dict, Union
from datetime import datetime
from protocol import ReportingRequest, ReportingResult
from tools.tools import generate_financial_report
from smolagents import CodeAgent, tool, OpenAIServerModel
from dotenv import load_dotenv
from json_repair import repair_json

load_dotenv()

# --- Tool Definitions ---
@tool
def generate_financial_report_tool(as_of_date: str = "2024-01-31") -> Dict:
    """
    Generate a comprehensive financial report for the company as of a specific date.
    
    Args:
        as_of_date (str): ISO-formatted date (YYYY-MM-DD) for the report cutoff
        
    Returns:
        Dict: Financial report containing cash balance, inventory, and analysis
    """
    try:
        return generate_financial_report(as_of_date)
    except Exception as e:
        return {"error": f"Failed to generate financial report: {str(e)}"}

SYSTEM_INSTRUCTIONS = """
        You are the ReportingAgent responsible for generating a financial summary of the company as of a given date.

Your tasks are:

1. Parse the requested date from the user query. It may appear as natural language, e.g., "as of June 30" or "today".

2. Normalize this date using standard Python `datetime` handling. If ambiguous, use `datetime.today()`.

3. Call the tool `generate_financial_report(as_of_date)` with the normalized date string.

4. Interpret the tool response and extract these fields:
   - as_of_date
   - cash_balance
   - inventory_value
   - total_assets
   - inventory_summary (list of items with stock/value)
   - top_selling_products (list of best performers by revenue)

5. Construct a final dictionary named `final_result` with:
   - **success**: True if report generated successfully, False if there was an error
   - **summary_text**: a human-readable paragraph summarizing key insights
   - **key_metrics**: a dict containing `cash_balance`, `inventory_value`, `total_assets`
   - **top_selling_products**: a list of up to 5 dicts with `item_name`, `total_units`, `total_revenue`
   - **inventory_overview**: a list of dicts with `item_name`, `stock`, `value` (omit `unit_price` to simplify)

6. Print the `final_result` using standard Python `print(final_result)`.

### Example:
```python
final_result = {
    "success": True or False, # True if report generated successfully, False if there was an error
    "summary_text": "As of June 30, the company holds $12,500 in cash and $3,420.75 in inventory. Total assets amount to $15,920.75. The top-selling product is Cardstock with $450 in revenue. Inventory is concentrated in A4 paper and envelopes.",
    "key_metrics": {
        "cash_balance": 12500.00,
        "inventory_value": 3420.75,
        "total_assets": 15920.75
    },
    "top_selling_products": [
        {"item_name": "Cardstock", "total_units": 3000, "total_revenue": 450.0},
        {"item_name": "A4 paper", "total_units": 5000, "total_revenue": 250.0}
    ],
    "inventory_overview": [
        {"item_name": "A4 paper", "stock": 300, "value": 15.0},
        {"item_name": "Cardstock", "stock": 100, "value": 15.0}
    ]
}
print(final_result)

        """

class ReportingAgent:
    """
    Agent responsible for generating financial reports and analytics.
    Uses generate_financial_report tool to create comprehensive reports.
    """
    
    def __init__(self):
        """
        Initialize the ReportingAgent with a CodeAgent for financial analysis.
        
        Args:
            model: The language model to use (optional, will use default if None)
        """
        self.model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )    
        
        # Initialize the CodeAgent with tools and instructions
        self.agent = CodeAgent(
            tools=[generate_financial_report_tool],
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            max_steps=20,
            verbosity_level=0,
            additional_authorized_imports=["json", "os", "typing", "datetime", "pandas", "time"]
        )
    
    def run(self, reporting_request: ReportingRequest) -> ReportingResult:
        """
        Process a reporting request and generate financial analysis.
        
        Args:
            reporting_request: ReportingRequest containing report parameters
            
        Returns:
            ReportingResult: Contains report data and analysis
        """
        try:
            # Format the request for the agent
            request_text = f"""
            Generate a financial report with the following parameters:
            - Report Type: {reporting_request.report_type}
            - Period: {reporting_request.period} <-- e.g., "last month", "Q1 2025", "04/05/25
            """
            
            if reporting_request.filters:
                request_text += f"\n- Filters: {reporting_request.filters}"
            
            # Run the agent
            agent_response = self.agent.run(request_text)
            
            # Parse the response
            result_data = self._parse_agent_response(agent_response)
            
            if result_data.get("success", False):
                return ReportingResult(
                    success=True,
                    report_data=result_data,
                    summary=result_data.get("text_summary", "Report generated successfully"),
                    error_message=None
                )
            else:
                return ReportingResult(
                    success=False,
                    report_data={},
                    summary="",
                    error_message=result_data.get("error_message", "Failed to generate report")
                )
                
        except Exception as e:
            return ReportingResult(
                success=False,
                report_data={},
                summary="",
                error_message=f"ReportingAgent error: {str(e)}"
            )
    
    def _parse_agent_response(self, response: Any) -> dict:
        """
        Parse the agent response to extract JSON data.
        
        Args:
            response: Raw response from the agent
            
        Returns:
            dict: Parsed response data
        """
        try:
            # Convert response to string if needed
            response_str = str(response)
            
            # Try to find JSON in the response
            start_idx = response_str.find('{')
            end_idx = response_str.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_str[start_idx:end_idx + 1]
                return json.loads(repair_json(json_str))
            else:
                # Fallback: create a basic response structure
                return {
                    "success": False,
                    "report_data": {},
                    "summary": "",
                    "error_message": "Could not parse agent response"
                }
                
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "report_data": {},
                "summary": "",
                "error_message": f"Invalid JSON in agent response: {str(e)}",
                "raw_response_excerpt": json_str[:500]  # limit for safety/log size
            }
        except Exception as e:
            return {
                "success": False,
                "report_data": {},
                "summary": "",
                "error_message": f"Response parsing error: {str(e)}",
                "raw_response_excerpt": json_str[:500]  # limit for safety/log size
            }
