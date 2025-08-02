# Beaver's Choice Paper Company Multi-Agent System

Welcome to the **Beaver's Choice Paper Company Multi-Agent System**! This project demonstrates a complete, production-ready multi-agent system that automates the entire customer quote-to-order workflow for a paper supply company.

## Project Overview

This system implements a sophisticated **5-agent orchestration** that handles:

- **Customer Request Parsing** - Intelligent extraction of items and delivery dates from natural language
- **Inventory Management** - Real-time stock checking with automated restocking capabilities  
- **Dynamic Quote Generation** - Pricing with bulk discounts and historical quote analysis
- **Order Processing** - Transaction creation and order finalization
- **Financial Reporting** - Comprehensive business analytics and reporting

The system processes natural language customer requests and produces complete business transactions, from initial inquiry to final order confirmation.

## Architecture

### 🤖 **5-Agent System**

1. **ParserAgent** - Extracts structured data from customer requests
2. **InventoryAgent** - Manages stock levels and procurement decisions
3. **QuoteAgent** - Calculates pricing with discounts and market analysis
4. **OrderAgent** - Finalizes sales transactions and generates order confirmations
5. **ReportingAgent** - Provides financial insights and business analytics

### 🔄 **Orchestration Flow**

```
Customer Request → Parse → Inventory Check → Quote Generation → Order Processing → Reporting
```

The **Orchestrator** coordinates all agents using a finite state machine with intelligent branching logic for handling partial fulfillment scenarios.

## Technology Stack

- **Framework**: `smolagents` for LLM-powered agent execution
- **Database**: SQLite with transaction tracking and inventory management
- **Data Processing**: `pandas` for financial calculations and reporting
- **Message Protocol**: `pydantic` models for type-safe inter-agent communication
- **State Management**: Custom finite state machine for workflow coordinationn Multi-Agent System Project

Welcome to the starter code repository for the **Beaver's Choice Paper Company Multi-Agent System Project**! This repository contains the starter code and tools you will need to design, build, and test a multi-agent system that supports core business operations at a fictional paper manufacturing company.

## Project Context

You’ve been hired as an AI consultant by Beaver's Choice Paper Company, a fictional enterprise looking to modernize their workflows. They need a smart, modular **multi-agent system** to automate:

- **Inventory checks** and restocking decisions
- **Quote generation** for incoming sales inquiries
- **Order fulfillment** including supplier logistics and transactions

Your solution must use a maximum of **5 agents** and process inputs and outputs entirely via **text-based communication**.

This project challenges your ability to orchestrate agents using modern Python frameworks like `smolagents`, `pydantic-ai`, or `npcsh`, and combine that with real data tools like `sqlite3`, `pandas`, and LLM prompt engineering.

## Key Features

### 🎯 **Intelligent Processing**
- **Natural Language Understanding**: Processes customer requests in plain English
- **Smart Pricing**: Implements bulk discount logic (10% for ≥100 units, 15% for ≥500 units)
- **Historical Analysis**: Leverages past quotes for better pricing decisions
- **Partial Fulfillment**: Gracefully handles scenarios where only some items are available

### 💾 **Robust Data Management**
- **Real-time Inventory**: Live stock tracking with automatic restocking triggers
- **Transaction Logging**: Complete audit trail of all business operations
- **Financial Reporting**: Automated generation of business insights and analytics
- **Type-safe Communication**: Pydantic models ensure data integrity across agents

### 🔧 **Production Features**
- **Error Recovery**: Comprehensive error handling and graceful fallbacks
- **Logging**: Detailed logging for debugging and monitoring
- **Testing Suite**: Complete test coverage for all agents and integration scenarios
- **Database Management**: Automated database initialization and cleanup

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or compatible LLM endpoint)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd beavers-choice-paper
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
Create a `.env` file with your API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Usage

**Initialize database and process a request:**
```bash
python main.py --init-db --request "I need 100 sheets of A4 paper and 50 envelopes for delivery by August 15th"
```

**Process request from file:**
```bash
python main.py --file customer_request.txt
```

**Test individual agents:**
```bash
python tests/test_inventory_agent.py
python tests/test_quote_agent.py
python tests/test_order_agent.py
```

## Project Structure

```
beavers-choice-paper/
├── orchestrator.py              # Main orchestration engine
├── main.py                      # CLI interface
├── agents/                      # Individual agent implementations
│   ├── parser_agent.py         # Customer request parsing
│   ├── inventory_agent.py      # Stock management
│   ├── quote_agent.py          # Pricing and quotes
│   ├── order_agent.py          # Order processing
│   ├── reporting_agent.py      # Financial reporting
│   └── message_protocol.py     # Type-safe communication models
├── framework/                   # Core framework components
│   ├── state_machine.py        # Workflow orchestration
├── tools/                       # Database and business logic
│   └── tools.py                # Database operations and utilities
├── data/                        # Sample data and test cases
│   ├── quotes.csv              # Historical quote data
│   ├── quote_requests.csv      # Customer request samples
│   └── quote_requests_sample.csv
├── tests/                       # Comprehensive test suite
│   ├── test_inventory_agent.py
│   ├── test_quote_agent.py
│   ├── test_order_agent.py
│   └── test_integration.py
└── Project Notebook.ipynb      # Development documentation
```

## Example Usage

### Basic Quote Request
```bash
$ python main.py --init-db --request "I need 200 sheets of A4 paper for my office"

🔧 Initializing database...
✅ Database ready.

📝 Processing request: I need 200 sheets of A4 paper for my office...

🚀 Running multi-agent workflow...

🧾 Final Customer Response:

Order placed successfully. Your quote for 200 sheets of A4 paper totals $9.00 
(including 10% bulk discount). Order ID: ORD-1722441600. 
Estimated delivery: August 15, 2025.

💰 Quote Total: $9.00
```

### Complex Multi-Item Request
```bash
$ python main.py --request "I need 150 A4 sheets, 50 envelopes, and 25 cardstock sheets for a wedding on August 20th"

# Handles multiple items, checks inventory, applies discounts, creates transactions
```

## Message Protocol

The system uses type-safe Pydantic models for all inter-agent communication:

```python
# Customer request parsing
ParserResult(
    items=[QuoteItem(name="A4 paper", quantity=150, unit="sheets")],
    delivery_date="2025-08-20",
    status="success"
)

# Inventory verification  
InventoryCheckResult(
    fulfillable_items=[...],
    unfulfillable_items=[...]
)

# Quote generation with pricing
QuoteResult(
    total_price=12.75,
    currency="USD", 
    line_items=[QuoteItem(name="A4 paper", quantity=150, unit_price=0.05, 
                         discount_percent=10.0, subtotal=6.75)],
    notes="Bulk discount applied"
)
```

## Testing

The project includes comprehensive testing for all components:

### Individual Agent Tests
```bash
# Test inventory management
python tests/test_inventory_agent.py

# Test quote generation with pricing
python tests/test_quote_agent.py  

# Test order processing and transactions
python tests/test_order_agent.py

# Test complete workflow integration
python tests/test_integration.py
```

### Integration Testing
```bash
# Test the complete orchestrator
python test_orchestrator.py
```

## Database

The system uses SQLite for data persistence with the following tables:

- **inventory**: Product catalog with pricing and stock levels
- **transactions**: Complete audit log of all business operations  
- **quotes**: Historical quote data for analysis
- **quote_requests**: Customer request archive

Database is automatically initialized with sample data when using the `--init-db` flag.

## Key Design Decisions

### Agent Architecture
- **Stateless Agents**: Each agent is stateless and communicates via structured messages
- **Type Safety**: Pydantic models ensure data integrity throughout the workflow
- **Tool Integration**: Agents use specialized tools for database operations and business logic

### Workflow Management  
- **Finite State Machine**: Orchestrator uses FSM for reliable workflow coordination
- **Branching Logic**: Intelligent handling of partial fulfillment scenarios
- **Error Recovery**: Graceful fallbacks for various failure modes

### Business Logic
- **Dynamic Pricing**: Bulk discounts based on quantity thresholds
- **Historical Analysis**: Past quote data influences current pricing decisions
- **Inventory Management**: Automated restocking triggers based on demand

## Contributing

This project demonstrates production-ready multi-agent system patterns. Key areas for extension:

- **Additional Agents**: Implement customer service or shipping agents
- **Enhanced Analytics**: More sophisticated financial reporting and forecasting
- **External Integrations**: Connect to real inventory systems or payment processors
- **Advanced Pricing**: Machine learning-based dynamic pricing models

## License

This project is provided as an educational example of multi-agent system architecture and implementation.

---

**Built with ❤️ using smolagents, SQLite, and modern Python patterns**