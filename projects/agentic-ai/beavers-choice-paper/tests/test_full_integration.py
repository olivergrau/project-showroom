# set cwd directory to the root of the project
import os
import sys
from pathlib import Path
import traceback

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
# MULTI AGENT SYSTEM INTEGRATION TEST
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
        verbosity=1  # Enable domain-specific prints for integration test
    )


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    """
    Run integration test scenarios using quote_requests_sample.csv data.
    Tests the complete orchestrator pipeline for each customer request.
    """
    print("🏗️  BEAVER'S CHOICE PAPER COMPANY - FULL INTEGRATION TEST")
    print("=" * 65)
    
    print("\n🔧 Initializing Database...")
    init_database(db_engine=db_engine, seed=0, debug=False)
    print("✅ Database initialized successfully")
    
    # Load test data
    try:
        # Try to load from parent directory (when run from tests/ folder)
        csv_path = "data/quote_requests_sample.csv"
        if not os.path.exists(csv_path):
            # Try loading from current directory (when run from project root)
            csv_path = "data/quote_requests_sample.csv"
        
        quote_requests_sample = pd.read_csv(csv_path)
        print(f"📂 Loaded {len(quote_requests_sample)} test scenarios from {csv_path}")
        
        # Parse dates
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
        
    except Exception as e:
        print(f"❌ FATAL: Error loading test data: {e}")
        return

    # Get initial financial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]
    
    print(f"\n💰 Initial Financial State ({initial_date}):")
    print(f"   Cash Balance: ${current_cash:.2f}")
    print(f"   Inventory Value: ${current_inventory:.2f}")
    print(f"   Total Assets: ${current_cash + current_inventory:.2f}")

    # Initialize orchestrator
    print("\n🤖 Initializing Multi-Agent Orchestrator...")
    try:
        # Test orchestrator creation
        test_orchestrator = create_orchestrator()
        print("✅ Orchestrator initialized with 5 agents")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return

    # Process each request through the orchestrator
    results = []
    successful_requests = 0
    failed_requests = 0
    
    print(f"\n🚀 Processing {len(quote_requests_sample)} customer requests...")
    print("-" * 65)
    
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")
        request_id = idx + 1

        print(f"\n📋 Request {request_id}/{len(quote_requests_sample)}")
        print(f"🏢 Context: {row['job']} organizing {row['event']}")
        print(f"📅 Date: {request_date}")
        print(f"💰 Cash: ${current_cash:.2f} | 📦 Inventory: ${current_inventory:.2f}")

        # Prepare request with context
        request_with_context = f"{row['request']} (Requested for {request_date})"
        print(f"📝 Request: {request_with_context[:80]}...")

        # Process request through orchestrator (create fresh instance for each request)
        try:
            print("🔄 Running orchestrator pipeline...")
            orchestrator = create_orchestrator()  # Fresh orchestrator for each request
            final_report = orchestrator.run(
                user_request=request_with_context,
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
            
            print(f"✅ Status: {response_status}")
            print(f"📝 Message: {response_message}")
            if quote_total > 0:
                print(f"💰 Quote: ${quote_total:.2f} for {quote_items} items")
            
            successful_requests += 1
            
        except Exception as e:
            response_summary = f"[ERROR] System error: {str(e)}"
            print(f"❌ Error processing request: {e}")
            traceback.print_exc()  # This prints the full stack trace to stderr
            failed_requests += 1

        # Update financial state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]
        overall_assets = current_cash + current_inventory

        print(f"📊 Updated: Cash ${current_cash:.2f} | Inventory ${current_inventory:.2f} | Total Assets ${overall_assets:.2f}")

        # Store results
        results.append({
            "request_id": request_id,
            "request_date": request_date,
            "job_context": row['job'],
            "event_context": row['event'],
            "original_request": row['request'],
            "response_summary": response_summary,
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "total_assets": current_cash + current_inventory,
            "quote_total": quote_total if 'quote_total' in locals() else 0.0,
            "processing_status": "success" if 'final_report' in locals() else "error"
        })

        # Brief pause between requests
        time.sleep(0.5)

    # Final financial report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report_data = generate_financial_report(final_date)
    
    print("\n" + "=" * 65)
    print("🏆 INTEGRATION TEST SUMMARY")
    print("=" * 65)
    print(f"📊 Total Requests Processed: {len(quote_requests_sample)}")
    print(f"✅ Successful: {successful_requests}")
    print(f"❌ Failed: {failed_requests}")
    print(f"📈 Success Rate: {(successful_requests/len(quote_requests_sample)*100):.1f}%")
    
    print(f"\n💰 Final Financial State ({final_date}):")
    print(f"   Cash Balance: ${final_report_data['cash_balance']:.2f}")
    print(f"   Inventory Value: ${final_report_data['inventory_value']:.2f}")
    print(f"   Total Assets: ${final_report_data['total_assets']:.2f}")
    
    # Calculate business metrics
    total_revenue = sum([r['quote_total'] for r in results if r['quote_total'] > 0])
    print(f"   Total Revenue Generated: ${total_revenue:.2f}")

    # Save detailed results
    results_df = pd.DataFrame(results)
    output_file = "integration_test_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    print("\n🎯 Integration test completed successfully!")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()