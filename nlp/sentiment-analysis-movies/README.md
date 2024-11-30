# Multilingual Movie Review Sentiment Analysis

## Project Overview

The objective of this project is to analyze the sentiment of movie reviews across three different languages: English, French, and Spanish. Using movie reviews and synopses from 30 movies (10 for each language), the project integrates natural language processing techniques to process, translate, and analyze sentiments.

The project involves the following steps:

1. **Data Integration**:
   - Read data from the provided `.csv` files: `movie_reviews_eng.csv`, `movie_reviews_fr.csv`, and `movie_reviews_sp.csv`.
   - Combine the data into a single pandas dataframe with the following columns: Title, Year, Synopsis, Review, and Original Language.

2. **Translation**:
   - Translate the French and Spanish reviews and synopses into English using pre-trained transformers from HuggingFace.

3. **Sentiment Analysis**:
   - Perform sentiment analysis on all reviews using HuggingFace pre-trained transformers.
   - Add a `Sentiment` column to the dataframe indicating whether each review is Positive or Negative.

4. **Output**:
   - Export the final dataframe to a CSV file. The resulting dataset will include columns for Title, Year, Synopsis, Review, Sentiment, and Original Language. The `Original Language` column will specify the original language of the review (`en/fr/sp`) before translation.

---

## Tools and Technologies

- **Pandas**: For data manipulation and integration.
- **HuggingFace Transformers**: For natural language processing tasks, including translation and sentiment analysis.
- **PyTorch**: For leveraging pre-trained machine learning models.

---

## Key Learning Outcomes

1. **Data Integration**:
   - Clean and combine multilingual data into a structured format using Pandas.

2. **Natural Language Processing**:
   - Translate text across multiple languages using pre-trained models.
   - Analyze text sentiment effectively using HuggingFace Transformers.

3. **Machine Learning**:
   - Apply pre-trained machine learning models for complex tasks such as translation and sentiment analysis.

4. **Workflow Integration**:
   - Combine multiple tools and libraries to solve a real-world problem involving multilingual data.

---

## Applications Demonstrated

- **Multilingual Data Processing**: Handle and process data across different languages.
- **Text Translation**: Use NLP models to bridge language barriers in data analysis.
- **Sentiment Analysis**: Extract insights into public perception based on textual data.
- **Data Output**: Generate a clean, structured CSV file for further analysis or reporting.

This project showcases the power of combining natural language processing, data analysis, and machine learning to extract meaningful insights from multilingual datasets.
