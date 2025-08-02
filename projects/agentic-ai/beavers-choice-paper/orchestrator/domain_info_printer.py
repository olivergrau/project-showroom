# domain_info_printer.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from protocol import QuoteResult, OrderResult, FinalReport

class DomainInfoPrinter(ABC):
    """Abstract base class for domain information printing."""
    
    @abstractmethod
    def print_step_info(self, step_name: str, state: Dict[str, Any], result: Dict = None):
        pass

class ConsoleDomainPrinter(DomainInfoPrinter):
    """Console-based domain information printer."""
    
    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity
    
    def print_step_info(self, step_name: str, state: Dict[str, Any], result: Dict = None):
        if self.verbosity == 0:
            return
            
        print(f"\n{'='*60}")
        print(f"📋 STEP: {step_name.upper()}")
        print(f"{'='*60}")
        
        printer_method = getattr(self, f"_print_{step_name}_info", None)
        if printer_method:
            printer_method(state, result)
        else:
            self._print_generic_info(step_name, state, result)
    
    def _print_parse_info(self, state: Dict[str, Any], result: Dict):
        """Print parsing step information."""
        print(f"📝 Original Request: {state['original_request']}")
        print(f"📅 Quote Request Date: {state['quote_request_date']}")
        
        if result and result.get('parsed_items'):
            items = result['parsed_items']
            print(f"✅ Successfully parsed {len(items)} items:")
            for item in items:
                print(f"   • {item.name}: {item.quantity} units")
            
            # print also unmatched items
            unmatched = result.get('unmatched_items', [])
            if unmatched:
                print(f"⚠️ Unmatched Items: {len(unmatched)}")
                for item in unmatched:
                    print(f"   • {item}")
            else:
                print("   No unmatched items found.")

            delivery_date = result.get('delivery_date')
            if delivery_date:
                print(f"🚚 Requested Delivery Date: {delivery_date}")
            else:
                print("⚠️ No specific delivery date requested")
        else:
            print("❌ Failed to parse any items from request")
    
    def _print_check_inventory_info(self, state: Dict[str, Any], result: Dict):
        """Print inventory check information."""
        self._print_financial_position(state['quote_request_date'])
        self._print_inventory_analysis(result)
        self._print_fulfillment_status(result)
    
    def _print_generate_quote_info(self, state: Dict[str, Any], result: Dict):
        """Print quote generation information."""
        if result and result.get('quote_result'):
            quote = result['quote_result']
            self._print_quote_summary(quote)
            self._print_quote_breakdown(quote)
        else:
            print("❌ Failed to generate quote")
    
    def _print_finalize_order_info(self, state: Dict[str, Any], result: Dict):
        """Print order finalization information."""
        if result and result.get('order_result'):
            order = result['order_result']
            self._print_order_status(order)
            if order.success:
                self._print_financial_impact(state, order)
        else:
            print("❌ Failed to process order")
    
    def _print_reporting_info(self, state: Dict[str, Any], result: Dict):
        """Print final reporting information."""
        if result and result.get('final_report'):
            report = result['final_report']
            self._print_business_summary(report)
    
    def _print_decline_request_info(self, state: Dict[str, Any], result: Dict):
        """Print decline information."""
        print(f"❌ Request Declined:")
        print(f"   Reason: Unable to fulfill customer requirements")
        
        if state.get('parsed_items'):
            print(f"   Requested Items: {len(state['parsed_items'])}")
            for item in state['parsed_items']:
                print(f"   • {item.name}: {item.quantity} units")
        
        # Include restockable items information if available
        restockable = state.get('restockable_items', [])
        if restockable:
            print(f"   🔄 Restockable Options: {len(restockable)} items could be ordered from suppliers")
            for item_name in restockable:
                print(f"   • {item_name}: Available for restocking")
        
        print(f"   📝 Customer will receive: Decline notification")
    
    # Helper methods for detailed printing
    def _print_financial_position(self, quote_request_date: str):
        from tools.tools import generate_financial_report
        financial_state = generate_financial_report(quote_request_date)
        
        print(f"💰 Current Financial Position:")
        print(f"   Cash Balance: ${financial_state['cash_balance']:,.2f}")
        print(f"   Inventory Value: ${financial_state['inventory_value']:,.2f}")
        print(f"   Total Assets: ${financial_state['total_assets']:,.2f}")
    
    def _print_inventory_analysis(self, result: Dict):
        if not result:
            return
            
        fulfillable = result.get('fulfillable_items', [])
        unfulfillable = result.get('unfulfillable_items', [])
        restockable = result.get('restockable_items', [])
        
        print(f"\n📦 Inventory Analysis:")
        print(f"   ✅ Fulfillable Items: {len(fulfillable)}")
        print(f"   ❌ Unfulfillable Items: {len(unfulfillable)}")
        if restockable:
            print(f"   🔄 Restockable Items: {len(restockable)}")
        
        if fulfillable:
            print(f"\n✅ Items Available for Fulfillment:")
            for item in fulfillable:
                print(f"   • {item.name}: {item.quantity} units")
        
        if unfulfillable:
            print(f"\n❌ Items NOT Available:")
            for item in unfulfillable:
                print(f"   • {item.name}: {item.quantity} units (out of stock or delivery issues)")
        
        if restockable:
            print(f"\n🔄 Items Available for Restocking:")
            for item_name in restockable:
                print(f"   • {item_name}: Can be restocked from suppliers")
    
    def _print_fulfillment_status(self, result: Dict):
        restockable = result.get('restockable_items', [])
        
        if result.get('all_items_fulfillable'):
            print(f"\n🎯 Status: ALL ITEMS AVAILABLE - Full order possible")
        elif result.get('some_items_fulfillable'):
            print(f"\n⚠️ Status: PARTIAL FULFILLMENT - Some items unavailable")
            if restockable:
                print(f"   📋 Note: {len(restockable)} items can be restocked if needed")
        else:
            print(f"\n❌ Status: ORDER DECLINED - No items available")
            if restockable:
                print(f"   📋 Note: {len(restockable)} items could be restocked with supplier order")
    
    def _print_quote_summary(self, quote: QuoteResult):
        print(f"💰 Quote Generated:")
        print(f"   Total Amount: ${quote.total_price:,.2f} {quote.currency}")
        print(f"   Line Items: {len(quote.line_items)}")
    
    def _print_quote_breakdown(self, quote: QuoteResult):
        if quote.line_items:
            print(f"\n📋 Quote Breakdown:")
            total_quantity = 0
            for item in quote.line_items:
                print(f"   • {item.name}: {item.quantity} units @ ${item.unit_price:.2f}/unit")
                if hasattr(item, 'discount_percent') and item.discount_percent > 0:
                    print(f"     Discount: {item.discount_percent}%")
                print(f"     Subtotal: ${item.subtotal:.2f}")
                total_quantity += item.quantity
            
            print(f"\n📊 Quote Summary:")
            print(f"   Total Quantity: {total_quantity} units")
            print(f"   Average Price per Unit: ${quote.total_price/total_quantity:.2f}")
        
        if quote.notes:
            print(f"\n📝 Notes: {quote.notes}")
    
    def _print_order_status(self, order: OrderResult):
        print(f"📋 Order Processing:")
        if order.success:
            print(f"   ✅ Order Status: SUCCESSFUL")
            print(f"   🔢 Order ID: {order.order_id}")
        else:
            print(f"   ❌ Order Status: FAILED")
        
        if order.message:
            print(f"   📝 Message: {order.message}")
    
    def _print_financial_impact(self, state: Dict[str, Any], order: OrderResult):
        from tools.tools import generate_financial_report
        financial_after = generate_financial_report(state['quote_request_date'])
        quote_total = getattr(state.get('quote_result', {}), 'total_price', 0) or 0
        
        print(f"\n💰 Financial Impact:")
        print(f"   Revenue Generated: ${quote_total:,.2f}")
        print(f"   Updated Cash Balance: ${financial_after['cash_balance']:,.2f}")
        print(f"   Updated Inventory Value: ${financial_after['inventory_value']:,.2f}")
        print(f"   Updated Total Assets: ${financial_after['total_assets']:,.2f}")
    
    def _print_business_summary(self, report: FinalReport):
        print(f"📊 Final Business Report:")
        print(f"   Status: {report.status.upper()}")
        print(f"   Message: {report.message}")
        
        if report.quote:
            print(f"\n💼 Business Metrics:")
            print(f"   Order Value: ${report.quote.total_price:,.2f}")
            print(f"   Items Sold: {len(report.quote.line_items)}")
            
            if report.status == "success":
                print(f"   ✅ Transaction: COMPLETED")
            elif report.status == "partial":
                print(f"   ⚠️ Transaction: PARTIAL FULFILLMENT")
        
        if report.unfulfillable_items:
            print(f"\n❌ Unfulfilled Items: {len(report.unfulfillable_items)}")
            for item in report.unfulfillable_items:
                print(f"   • {item.name}: {item.quantity} units")
    
    def _print_generic_info(self, step_name: str, state: Dict[str, Any], result: Dict):
        print(f"ℹ️ Step '{step_name}' completed")
        if result:
            print(f"   Result keys: {list(result.keys())}")

class SilentDomainPrinter(DomainInfoPrinter):
    """Silent printer that outputs nothing."""
    
    def print_step_info(self, step_name: str, state: Dict[str, Any], result: Dict = None):
        pass
