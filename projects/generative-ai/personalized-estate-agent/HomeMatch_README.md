
# HomeMatch RealEstate Agent

HomeMatch is an intelligent real estate recommendation system designed to match user preferences with the most suitable property listings. The project uses machine learning techniques and a vector database to perform semantic searches, apply metadata filters, and personalize recommendations based on user input.

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Features
- **Semantic Search**: Matches user preferences with property listings based on textual similarity using embeddings.
- **Metadata Filtering**: Filters listings by bedrooms, bathrooms, price, and more.
- **Personalized Recommendations**: Enhances listing descriptions to align with user preferences.
- **Interactive Workflow**: Implemented in Jupyter Notebook for easy experimentation and execution.

---

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- Virtual environment tool (optional but recommended)

Dependencies:
- `langchain`
- `chromadb`
- `openai`
- `sentence-transformers`
- `pandas`
- `numpy`

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/homematch-realestate.git
   cd homematch-realestate
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the OpenAI API key:
   - Create a `.env` file in the project directory and add your OpenAI API key:
     ```bash
     OPENAI_API_KEY=your_openai_api_key
     ```

5. (Optional) Verify installation:
   ```bash
   pip list
   ```

---

## How to Run

1. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the notebook file:
   - Navigate to `HomeMatch.ipynb` in your Jupyter Notebook interface.

3. Follow the step-by-step instructions provided in the notebook to:
   - Generate embeddings for property listings.
   - Store and query listings using ChromaDB.
   - Personalize descriptions based on user preferences.

4. View the final recommendations and interact with the results.

---

## Usage

### Key Functionalities:
- **Load Listings**:
  Load property data (e.g., `real_estate_listings.csv`) and preprocess for vector database storage.

- **Generate Embeddings**:
  Use `sentence-transformers` to create semantic embeddings for the property descriptions.

- **Store in ChromaDB**:
  Persist embeddings and metadata in ChromaDB for efficient querying.

- **Search Listings**:
  Query the vector database using user preferences. Apply metadata filters like:
  - Number of bedrooms
  - Number of bathrooms
  - Price range

- **Personalized Recommendations**:
  Leverage a language model to refine and personalize the output listings.

### Example Workflow:
1. Load property data.
2. Generate embeddings for property descriptions.
3. Apply user filters (e.g., `{"bedrooms": {"$gte": 3}, "price": {"$lte": 500000}}`).
4. Query the database and view results.

---

## Technologies Used
- **Programming Language**: Python 3.9+
- **Machine Learning**:
  - `sentence-transformers` for generating embeddings
- **Database**:
  - `chromadb` for vector-based storage and search
- **NLP**:
  - OpenAI API for personalized recommendations
- **Data Analysis**:
  - `pandas`, `numpy`
- **Interactive Environment**:
  - Jupyter Notebook