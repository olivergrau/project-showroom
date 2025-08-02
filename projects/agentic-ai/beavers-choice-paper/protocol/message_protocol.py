# protocol/message_protocol.py

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel

class QuoteItem(BaseModel):
    name: str
    quantity: int
    
    # Pricing and discount information (optional, used in QuoteResult line_items)
    unit_price: Optional[float] = None
    discount_percent: Optional[float] = None
    subtotal: Optional[float] = None

class ParserResult(BaseModel):
    items: List[QuoteItem]
    delivery_date: str
    unmatched_items: List[str] = []
    status: Literal["success", "partial", "declined"] = "success"

class InventoryCheckResult(BaseModel):
    fulfillable_items: List[QuoteItem]
    unfulfillable_items: List[QuoteItem]
    restockable_items: List[str] = []

class QuoteResult(BaseModel):
    success: bool = False
    total_price: float
    currency: str
    line_items: List[QuoteItem]
    notes: Optional[str] = None

class OrderResult(BaseModel):
    success: bool
    order_id: Optional[str]
    message: Optional[str]

class ReportingRequest(BaseModel):
    report_type: Literal["financial", "inventory", "sales", "custom"]
    period: str  # e.g., "last_30_days", "2024-01", "Q1_2024"
    filters: Optional[Dict] = None  # Additional filters for the report

class ReportingResult(BaseModel):
    success: bool
    report_data: Dict
    summary: Optional[str] = None
    error_message: Optional[str] = None

class FinalReport(BaseModel):
    status: Literal["success", "partial", "declined"]
    report: Optional[ReportingResult] = None
    message: str
    quote: Optional[QuoteResult] = None
    unfulfillable_items: Optional[List[QuoteItem]] = None
    error_log: Optional[List[str]] = None
