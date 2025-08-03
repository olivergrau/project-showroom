# step_handlers.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any

from framework.state_machine import StateMachine
from protocol import *

class StepHandler(ABC):
    """Abstract base class for step handlers."""
    
    def __init__(self, agent_manager, domain_printer):
        self.agent_manager = agent_manager
        self.domain_printer = domain_printer
    
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def _print_domain_info(self, step_name: str, state: Dict[str, Any], result: Dict):
        self.domain_printer.print_step_info(step_name, state, result)

class ParseStepHandler(StepHandler):
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result: ParserResult = self.agent_manager.parser_agent.run(
            quote_request=state["original_request"],
            quote_request_date=state["quote_request_date"]
        )
        
        step_result = {
            "parsed_items": result.items,
            "delivery_date": result.delivery_date
        }
        
        self._print_domain_info("parse", state, step_result)
        return step_result

class InventoryStepHandler(StepHandler):
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        inv: InventoryCheckResult = self.agent_manager.inventory_agent.run(
            quote_items=state["parsed_items"], 
            delivery_date=state["delivery_date"],
            quote_request_date=state["quote_request_date"]
        )
        
        f = inv.fulfillable_items
        u = inv.unfulfillable_items
        
        step_result = {
            "fulfillable_items": f,
            "unfulfillable_items": u,
            "all_items_fulfillable": len(f) == len(state["parsed_items"]),
            "some_items_fulfillable": 0 < len(f) < len(state["parsed_items"]),
            "no_items_fulfillable": len(f) == 0,
            "restockable_items": inv.restockable_items,
            "stock_orders": inv.stock_orders,
        }
        
        self._print_domain_info("check_inventory", state, step_result)
        return step_result

class QuoteStepHandler(StepHandler):
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        qr: QuoteResult = self.agent_manager.quote_agent.run(state["fulfillable_items"])
        
        step_result = {"quote_result": qr}
        self._print_domain_info("generate_quote", state, step_result)
        return step_result

class OrderStepHandler(StepHandler):
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        orr: OrderResult = self.agent_manager.order_agent.run(
            quote_result=state["quote_result"],
            quote_request_date=state["quote_request_date"]
        )
        
        step_result = {"order_result": orr}
        self._print_domain_info("finalize_order", state, step_result)
        return step_result

class ReportingStepHandler(StepHandler):
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        reporting_result = None
        
        if state["no_items_fulfillable"]:
            status = "declined"
            msg = "Unable to fulfill any items."
        elif state["some_items_fulfillable"]:
            status = "partial"
            msg = "Partially fulfilled. See order details."
        else:
            status = "success"
            msg = "Order placed successfully."
            financial_request = ReportingRequest(
                report_type="financial",
                period=state["quote_request_date"],
                filters={"include_trends": True}
            )
            reporting_result = self.agent_manager.reporting_agent.run(
                reporting_request=financial_request                
            )
        
        final = FinalReport(
            status=status,
            message=msg,
            report=reporting_result,
            quote=state["quote_result"] if status != "declined" else None,
            unfulfillable_items=state["unfulfillable_items"] or None,
            error_log=None
        )
        
        step_result = {"final_report": final}
        self._print_domain_info("reporting", state, step_result)
        return step_result

class DeclineStepHandler(StepHandler):
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        final = FinalReport(
            status="declined",
            message="We're sorry — your request cannot be fulfilled at this time.",
            quote=None,
            unfulfillable_items=state["parsed_items"],
            error_log=None
        )
        
        step_result = {"final_report": final}
        self._print_domain_info("decline_request", state, step_result)
        return step_result
