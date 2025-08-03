# orchestrator.py (refactored)

import logging
from typing import List, Dict, Any

from framework.state_machine import Step, EntryPoint, Termination, StateMachine
from protocol import *
from .domain_info_printer import ConsoleDomainPrinter, SilentDomainPrinter, DomainInfoPrinter
from .agent_manager import AgentManager
from .step_handlers import *

class OrchestratorState(Dict[str, Any]):
    original_request: str
    quote_request_date: str
    delivery_date: str
    parsed_items: List[QuoteItem]
    fulfillable_items: List[QuoteItem]
    unfulfillable_items: List[QuoteItem]
    all_items_fulfillable: bool
    some_items_fulfillable: bool
    no_items_fulfillable: bool
    restockable_items: List[str]
    stock_orders: List[Dict]
    quote_result: QuoteResult
    order_result: OrderResult
    final_report: FinalReport

class Orchestrator:
    """Clean, SOLID-compliant orchestrator for the multi-agent workflow."""
    
    def __init__(self, parser_agent, inventory_agent, quote_agent, order_agent,
                 reporting_agent, log_level: str = "info", verbosity: int = 1):
        
        # Dependency injection
        self.agent_manager = AgentManager(
            parser_agent, inventory_agent, quote_agent, 
            order_agent, reporting_agent, verbosity
        )
        
        self.domain_printer = self._create_domain_printer(verbosity)
        self.step_handlers = self._create_step_handlers()
        
        # Logging setup
        self.logger = self._setup_logger(log_level)
        
        # Build workflow
        self.workflow = self._build_state_machine()
    
    def _create_domain_printer(self, verbosity: int) -> DomainInfoPrinter:
        """Factory method for domain printer (Open/Closed Principle)."""
        if verbosity == 0:
            return SilentDomainPrinter()
        else:
            return ConsoleDomainPrinter(verbosity)
    
    def _create_step_handlers(self) -> Dict[str, StepHandler]:
        """Factory method for step handlers (Open/Closed Principle)."""
        return {
            "parse": ParseStepHandler(self.agent_manager, self.domain_printer),
            "check_inventory": InventoryStepHandler(self.agent_manager, self.domain_printer),
            "generate_quote": QuoteStepHandler(self.agent_manager, self.domain_printer),
            "finalize_order": OrderStepHandler(self.agent_manager, self.domain_printer),
            "reporting": ReportingStepHandler(self.agent_manager, self.domain_printer),
            "decline_request": DeclineStepHandler(self.agent_manager, self.domain_printer),
        }
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger configuration."""
        logger = logging.getLogger("Orchestrator")
        logger.propagate = False
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if log_level == "debug" else logging.INFO)
        
        return logger
    
    def _build_state_machine(self) -> StateMachine[OrchestratorState]:
        """Build the state machine workflow."""
        sm = StateMachine(state_schema=OrchestratorState, logger=self.logger)

        # Create steps using step handlers
        entry = EntryPoint()
        parse = Step("parse", self._step_parse)
        inventory = Step("check_inventory", self._step_check_inventory)
        quote = Step("generate_quote", self._step_generate_quote)
        order = Step("finalize_order", self._step_finalize_order)
        report = Step("reporting", self._step_reporting)
        decline = Step("decline_request", self._step_decline)
        terminate = Termination()

        sm.add_steps([entry, parse, inventory, quote, order, report, decline, terminate])

        # Define workflow connections
        sm.connect(entry, parse)
        sm.connect(parse, [inventory, decline], condition=self._decide_post_parsing)
        sm.connect(inventory, [quote, decline], condition=self._decide_post_inventory)
        sm.connect(quote, order)
        sm.connect(order, report)
        sm.connect(decline, report)
        sm.connect(report, terminate)

        return sm
    
    def run(self, user_request: str, quote_request_date: str = "2025-08-01") -> FinalReport:
        """
        Execute the orchestration workflow.
        
        Args:
            user_request: Customer's request to process
            quote_request_date: Date when quote was requested (ISO format)
            
        Returns:
            FinalReport: Final report from the orchestration process
        """
        initial_state: OrchestratorState = self._create_initial_state(
            user_request, quote_request_date)
        
        run = self.workflow.run(initial_state)
        final_state = run.get_final_state()
        
        return final_state["final_report"]
    
    def _create_initial_state(self, user_request: str, quote_request_date: str) -> OrchestratorState:
        """Create initial state with default values."""
        return {
            "original_request": user_request,
            "quote_request_date": quote_request_date,
            "delivery_date": None,
            "parsed_items": [],
            "fulfillable_items": [],
            "unfulfillable_items": [],
            "all_items_fulfillable": False,
            "some_items_fulfillable": False,
            "no_items_fulfillable": False,
            "restockable_items": [],
            "stock_orders": [],
            "quote_result": QuoteResult(success=False, total_price=0, currency="USD", line_items=[], notes=None),
            "order_result": OrderResult(success=False, order_id=None, message=None),
            "final_report": FinalReport(status="declined", message="", quote=None, 
                                      unfulfillable_items=None, error_log=None)
        }
    
    # Step implementations (delegate to handlers)
    def _step_parse(self, state: OrchestratorState) -> Dict:
        return self.step_handlers["parse"].execute(state)
    
    def _step_check_inventory(self, state: OrchestratorState) -> Dict:
        return self.step_handlers["check_inventory"].execute(state)
    
    def _step_generate_quote(self, state: OrchestratorState) -> Dict:
        return self.step_handlers["generate_quote"].execute(state)
    
    def _step_finalize_order(self, state: OrchestratorState) -> Dict:
        return self.step_handlers["finalize_order"].execute(state)
    
    def _step_reporting(self, state: OrchestratorState) -> Dict:
        return self.step_handlers["reporting"].execute(state)
    
    def _step_decline(self, state: OrchestratorState) -> Dict:
        return self.step_handlers["decline_request"].execute(state)
    
    # Decision logic
    def _decide_post_parsing(self, state: OrchestratorState) -> str:
        return "decline_request" if not state["parsed_items"] else "check_inventory"
    
    def _decide_post_inventory(self, state: OrchestratorState) -> str:
        return "decline_request" if state["no_items_fulfillable"] else "generate_quote"
