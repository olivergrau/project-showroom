# main.py

import argparse
import os
import glob
from orchestrator import Orchestrator

from agents.parser_agent import QuoteRequestParserAgent
from agents.inventory_agent import InventoryAgent
from agents.quote_agent import QuoteAgent          
from agents.order_agent import OrderAgent          
from agents.reporting_agent import ReportingAgent  

from tools.tools import db_engine, init_database

def load_quote_request(args) -> str:
    if args.request:
        return args.request.strip()
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"File not found: {args.file}")
        except Exception as e:
            raise ValueError(f"Error reading file {args.file}: {e}")
    else:
        raise ValueError("You must provide a quote request via --request or --file")

def main():
    parser = argparse.ArgumentParser(description="Process a quote request with agentic workflow.")
    parser.add_argument("--request", type=str, help="Quote request as plain text")
    parser.add_argument("--file", type=str, help="Path to file containing quote request")
    parser.add_argument("--init-db", action="store_true", help="Initialize database with sample inventory")
    parser.add_argument("--request-date", type=str, help="Date of the quote request (MM/DD/YY)")
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2], 
                       help="Verbosity level: 0=silent, 1=domain info, 2=domain info + agent debug")

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        
        # Delete existing database files before creating new ones
        print("Cleaning up existing database files...\n")
        
        # Search for munder_difflin.db files in current directory and db folder
        db_patterns = [
            "munder_difflin.db",           # Root directory
            "db/munder_difflin.db",        # db folder
            "**/munder_difflin.db"         # Any subdirectory (recursive)
        ]
        
        deleted_files = []
        for pattern in db_patterns:
            for db_file in glob.glob(pattern, recursive=True):
                if os.path.exists(db_file):
                    try:
                        os.remove(db_file)
                        deleted_files.append(db_file)
                        print(f"🗑️  Deleted existing database: {db_file}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete {db_file}: {e}")
        
        if deleted_files:
            print(f"✅ Cleaned up {len(deleted_files)} database file(s)\n")
        else:
            print("ℹ️  No existing database files found to clean up\n")
        
        # Initialize the database engine and create sample inventory
        print("Initializing database and generating sample inventory...\n")
        init_database(db_engine=db_engine, seed=0, debug=True)

        print("Sample inventory generated successfully.\n")

    # Load input
    user_input = load_quote_request(args)
    
    # Get quote request date, default to today if not provided
    quote_request_date = args.request_date if args.request_date else "2025-08-01"
    
    # If date is in MM/DD/YY format, convert to YYYY-MM-DD
    if quote_request_date and "/" in quote_request_date:
        try:
            from datetime import datetime
            parsed_date = datetime.strptime(quote_request_date, "%m/%d/%y")
            quote_request_date = parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Warning: Could not parse date '{quote_request_date}', using default")
            quote_request_date = "2025-08-01"

    # Initialize orchestrator with sub-agents (creating fresh instances)
    orchestrator = Orchestrator(
        parser_agent=QuoteRequestParserAgent(),         # ← Fresh agent instance
        inventory_agent=InventoryAgent(),   # ← Fresh agent instance
        quote_agent=QuoteAgent(),          # ← Fresh agent instance
        order_agent=OrderAgent(),          # ← Fresh agent instance
        reporting_agent=ReportingAgent(),   # ← Fresh agent instance
        verbosity=args.verbosity
    )

    # Run agentic workflow
    final_report = orchestrator.run(user_input, quote_request_date=quote_request_date)

    # Output final response
    print("\n🧾 Final Customer Response:\n")
    print(final_report.message)  # ✅ Access Pydantic model attribute

if __name__ == "__main__":
    main()
