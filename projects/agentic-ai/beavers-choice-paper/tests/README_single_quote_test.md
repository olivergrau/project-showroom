# Single Quote Request Test

## Usage

The `test_one_quote_request.py` script processes a single quote request through the complete orchestrator pipeline.

### Basic Usage

```bash
# Test with quote.txt and initialize database
python tests/test_one_quote_request.py quote.txt 2025-08-01

# Test without reinitializing database (use existing data)
python tests/test_one_quote_request.py --no-init-db quote.txt 2025-08-01

# Test with a different quote file
python tests/test_one_quote_request.py data/sample_quote.txt 2025-07-15
```

### Parameters

- `quote_file`: Path to the text file containing the quote request
- `request_date`: Date in YYYY-MM-DD format when the quote was requested
- `--no-init-db`: Optional flag to skip database initialization

### Features

✅ **Complete Pipeline Testing**: Tests all 5 agents (Parser, Inventory, Quote, Order, Reporting)
✅ **Financial Tracking**: Shows before/after financial state and changes
✅ **Detailed Output**: Comprehensive logging with emojis for easy reading
✅ **Error Handling**: Graceful error handling with detailed stack traces
✅ **Quote Details**: Shows line items, pricing, and totals when successful
✅ **CLI Interface**: Easy command-line usage with help and examples

### Output

The script provides:
- Initial financial state (cash, inventory, total assets)
- Quote processing results with detailed line items
- Final financial state with changes highlighted
- Processing status and response summary
- Business impact analysis

### Examples

```bash
# Process the included quote.txt for August 1st, 2025
cd /path/to/project
python tests/test_one_quote_request.py quote.txt 2025-08-01

# Use existing database data (no reinitialization)
python tests/test_one_quote_request.py --no-init-db quote.txt 2025-08-01
```

### Sample Output

```
🏗️  BEAVER'S CHOICE PAPER COMPANY - SINGLE QUOTE REQUEST TEST
======================================================================

🔧 Initializing Database...
✅ Database initialized successfully

📂 Loading quote request from: quote.txt
✅ Quote request loaded successfully
📝 Request preview: I would like to place an order for 500 sheets of colorful poster paper...

📅 Request Date: 2025-08-01

💰 Initial Financial State (2025-08-01):
   Cash Balance: $10000.00
   Inventory Value: $5234.50
   Total Assets: $15234.50

🤖 Initializing Multi-Agent Orchestrator...
✅ Orchestrator initialized with 5 agents

🚀 Processing quote request...
----------------------------------------------------------------------
🔄 Running orchestrator pipeline...

💰 Quote Details:
   Total Price: $157.50
   Line Items: 3
   1. Poster paper: 500 units @ $0.25 = $125.00
   2. Streamers: 300 units @ $0.08 = $24.00
   3. Balloons: 200 units @ $0.04 = $8.50

✅ Status: completed
📝 Message: Quote generated and order processed successfully

📊 Final Financial State (2025-08-01):
   Cash Balance: $10157.50 (change: +157.50)
   Inventory Value: $5077.00 (change: -157.50)
   Total Assets: $15234.50 (change: +0.00)

======================================================================
🏆 SINGLE QUOTE REQUEST TEST SUMMARY
======================================================================
📂 Quote File: quote.txt
📅 Request Date: 2025-08-01
🎯 Processing Status: SUCCESS
📝 Response: [COMPLETED] Quote generated and order processed successfully
💰 Quote Generated: $157.50 for 3 items
📈 Financial Impact:
   Cash: +157.50
   Inventory: -157.50
   Net Assets: +0.00

🎯 Single quote request test completed!
```
