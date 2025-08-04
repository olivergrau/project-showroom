# set cwd directory to the root of the project
import os
import sys
from pathlib import Path
import traceback
import argparse

# Get the absolute path to the current file (inside tests/)
current_file_path = Path(__file__).resolve()

# Set working directory to the parent of 'tests'
project_root = current_file_path.parent.parent
os.chdir(project_root)

# (Optional) Add root to sys.path if you want to import project modules directly
sys.path.insert(0, str(project_root))

print(f"Changed working directory to: {os.getcwd()}")

import pandas as pd
import numpy as np
import time
import dotenv
import sys
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Add parent directory to path to import project modules
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

from tools.tools import generate_financial_report, init_database, db_engine
from orchestrator import Orchestrator
from agents.parser_agent import QuoteRequestParserAgent
from agents.inventory_agent import InventoryAgent
from agents.quote_agent import QuoteAgent          
from agents.order_agent import OrderAgent          
from agents.reporting_agent import ReportingAgent

# Load environment variables
dotenv.load_dotenv()

########################
########################
########################
# SINGLE QUOTE REQUEST TEST
########################
########################
########################

def create_orchestrator():
    """
    Create and return an orchestrator instance with fresh agent instances.
    """
    return Orchestrator(
        parser_agent=QuoteRequestParserAgent(),
        inventory_agent=InventoryAgent(),
        quote_agent=QuoteAgent(),
        order_agent=OrderAgent(),
        reporting_agent=ReportingAgent(),
        verbosity=1  # Enable domain-specific prints for single quote test
    )


def load_quote_request(quote_file: str) -> str:
    """
    Load quote request from specified file.
    
    Args:
        quote_file (str): Path to the quote request file
        
    Returns:
        str: Content of the quote request file
        
    Raises:
        ValueError: If file cannot be read
    """
    try:
        with open(quote_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Quote file {quote_file} is empty")
        return content
    except FileNotFoundError:
        raise ValueError(f"Quote file not found: {quote_file}")
    except Exception as e:
        raise ValueError(f"Error reading quote file {quote_file}: {e}")


def test_single_quote_request(quote_file: str, request_date: str, init_db: bool = True):
    """
    Test a single quote request through the complete orchestrator pipeline.
    
    Args:
        quote_file (str): Path to the quote request file
        request_date (str): Date of the quote request in YYYY-MM-DD format
        init_db (bool): Whether to initialize the database with sample data
    """
    print("🏗️  BEAVER'S CHOICE PAPER COMPANY - SINGLE QUOTE REQUEST TEST")
    print("=" * 70)
    
    # Initialize database if requested
    if init_db:
        print("\n🔧 Initializing Database...")
        init_database(db_engine=db_engine, seed=0, debug=False)
        print("✅ Database initialized successfully")
    
    # Load quote request
    try:
        print(f"\n📂 Loading quote request from: {quote_file}")
        quote_request = load_quote_request(quote_file)
        print(f"✅ Quote request loaded successfully")
        print(f"📝 Request preview: {quote_request[:100]}{'...' if len(quote_request) > 100 else ''}")
    except ValueError as e:
        print(f"❌ FATAL: {e}")
        return None
    
    # Validate date format
    try:
        datetime.strptime(request_date, "%Y-%m-%d")
        print(f"📅 Request Date: {request_date}")
    except ValueError:
        print(f"❌ FATAL: Invalid date format. Expected YYYY-MM-DD, got: {request_date}")
        return None

    # Get initial financial state
    report = generate_financial_report(request_date)
    initial_cash = report["cash_balance"]
    initial_inventory = report["inventory_value"]
    initial_assets = initial_cash + initial_inventory
    
    print(f"\n💰 Initial Financial State ({request_date}):")
    print(f"   Cash Balance: ${initial_cash:.2f}")
    print(f"   Inventory Value: ${initial_inventory:.2f}")
    print(f"   Total Assets: ${initial_assets:.2f}")

    # Initialize orchestrator
    print("\n🤖 Initializing Multi-Agent Orchestrator...")
    try:
        orchestrator = create_orchestrator()
        print("✅ Orchestrator initialized with 5 agents")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return None

    # Process the quote request
    print(f"\n🚀 Processing quote request...")
    print("-" * 70)
    
    result = {
        "request_date": request_date,
        "quote_file": quote_file,
        "original_request": quote_request,
        "initial_cash": initial_cash,
        "initial_inventory": initial_inventory,
        "initial_assets": initial_assets,
        "processing_status": "unknown",
        "response_summary": "",
        "quote_total": 0.0,
        "quote_items": 0,
        "final_cash": 0.0,
        "final_inventory": 0.0,
        "final_assets": 0.0,
        "cash_change": 0.0,
        "inventory_change": 0.0,
        "assets_change": 0.0
    }

    try:
        print("🔄 Running orchestrator pipeline...")
        
        # Process request through orchestrator
        final_report = orchestrator.run(
            user_request=quote_request,
            quote_request_date=request_date
        )
        
        # Extract response details
        response_status = final_report.status
        response_message = final_report.message
        response_summary = f"[{response_status.upper()}] {response_message}"
        
        # Get quote details if available
        quote_total = 0.0
        quote_items = 0
        if final_report.quote:
            quote_total = final_report.quote.total_price
            quote_items = len(final_report.quote.line_items)
            
            print(f"\n💰 Quote Details:")
            print(f"   Total Price: ${quote_total:.2f}")
            print(f"   Line Items: {quote_items}")
            
            # Print line item details
            for i, item in enumerate(final_report.quote.line_items, 1):
                print(f"   {i}. {item.name}: {item.quantity} units @ ${item.unit_price:.2f}")
        
        print(f"\n✅ Status: {response_status}")
        print(f"📝 Message: {response_message}")
        
        result.update({
            "processing_status": "success",
            "response_summary": response_summary,
            "quote_total": quote_total,
            "quote_items": quote_items
        })
        
    except Exception as e:
        response_summary = f"[ERROR] System error: {str(e)}"
        print(f"❌ Error processing request: {e}")
        traceback.print_exc()
        
        result.update({
            "processing_status": "error",
            "response_summary": response_summary
        })

    # Get final financial state
    final_report_data = generate_financial_report(request_date)
    final_cash = final_report_data["cash_balance"]
    final_inventory = final_report_data["inventory_value"]
    final_assets = final_cash + final_inventory
    
    # Calculate changes
    cash_change = final_cash - initial_cash
    inventory_change = final_inventory - initial_inventory
    assets_change = final_assets - initial_assets
    
    result.update({
        "final_cash": final_cash,
        "final_inventory": final_inventory,
        "final_assets": final_assets,
        "cash_change": cash_change,
        "inventory_change": inventory_change,
        "assets_change": assets_change
    })

    print(f"\n📊 Final Financial State ({request_date}):")
    print(f"   Cash Balance: ${final_cash:.2f} (change: {cash_change:+.2f})")
    print(f"   Inventory Value: ${final_inventory:.2f} (change: {inventory_change:+.2f})")
    print(f"   Total Assets: ${final_assets:.2f} (change: {assets_change:+.2f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 SINGLE QUOTE REQUEST TEST SUMMARY")
    print("=" * 70)
    print(f"📂 Quote File: {quote_file}")
    print(f"📅 Request Date: {request_date}")
    print(f"🎯 Processing Status: {result['processing_status'].upper()}")
    print(f"📝 Response: {response_summary}")
    
    if quote_total > 0:
        print(f"💰 Quote Generated: ${quote_total:.2f} for {quote_items} items")
    
    if abs(cash_change) > 0.01 or abs(inventory_change) > 0.01:
        print(f"📈 Financial Impact:")
        if abs(cash_change) > 0.01:
            print(f"   Cash: {cash_change:+.2f}")
        if abs(inventory_change) > 0.01:
            print(f"   Inventory: {inventory_change:+.2f}")
        print(f"   Net Assets: {assets_change:+.2f}")
    else:
        print("📊 No financial transactions recorded")
    
    print("\n🎯 Single quote request test completed!")
    return result


def main():
    """
    Main function to handle command line arguments and run the test.
    """
    parser = argparse.ArgumentParser(
        description="Test a single quote request through the complete orchestrator pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_one_quote_request.py quote.txt 2025-08-01
  python test_one_quote_request.py --no-init-db data/sample_quote.txt 2025-07-15
        """
    )
    
    parser.add_argument("quote_file", type=str, 
                       help="Path to the quote request file (e.g., quote.txt)")
    parser.add_argument("request_date", type=str, 
                       help="Quote request date in YYYY-MM-DD format (e.g., 2025-08-01)")
    parser.add_argument("--no-init-db", action="store_true", 
                       help="Skip database initialization (use existing data)")
    
    args = parser.parse_args()
    
    # Run the test
    try:
        result = test_single_quote_request(
            quote_file=args.quote_file,
            request_date=args.request_date,
            init_db=not args.no_init_db
        )
        
        if result and result["processing_status"] == "success":
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
