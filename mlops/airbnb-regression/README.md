# Build an ML Pipeline for Short-Term Rental Prices in NYC

## Project Overview

This project focuses on building an end-to-end machine learning pipeline to estimate the typical price of short-term rental properties in New York City. The pipeline processes data, trains a model, and automates the workflow for retraining the model weekly as new data arrives. The project demonstrates how to create a reusable and scalable pipeline with a focus on MLops best practices.

The solution leverages tools like MLflow for pipeline orchestration, Weights & Biases (W&B) for experiment tracking and artifact storage, and Hydra for configuration management. By integrating these tools, the project automates key steps from data cleaning to model deployment.

---

## Key Features

1. **Exploratory Data Analysis (EDA)**:
   - Analyze the raw dataset to understand feature distributions and identify issues like missing values or outliers.
   - Create visualizations to gain insights into the data.

2. **Data Cleaning and Preprocessing**:
   - Handle missing values, remove outliers, and preprocess features for training.
   - Ensure data integrity with automated data checks.

3. **Feature Engineering**:
   - Transform features using techniques like scaling and encoding to prepare data for modeling.

4. **Model Training and Hyperparameter Tuning**:
   - Train a Random Forest model to predict rental prices.
   - Optimize hyperparameters using Hydra’s configuration system for reproducible experimentation.

5. **Pipeline Automation**:
   - Implement a modular pipeline with reusable components, enabling seamless retraining with new data.
   - Integrate data processing, model training, and evaluation steps.

6. **Model Evaluation and Deployment**:
   - Evaluate model performance using metrics like Mean Absolute Error (MAE).
   - Deploy the best-performing model to production and tag it as ready for use.

---

## Applications Demonstrated

- **Machine Learning Operations (MLops)**:
  - Automate end-to-end workflows for data handling, model training, and deployment.
  - Use tools like MLflow, Hydra, and W&B for efficient pipeline orchestration.

- **Data-Driven Decision Making**:
  - Predict rental prices accurately, helping property management companies set competitive rates.

- **Scalable and Reusable Solutions**:
  - Build a pipeline that adapts to weekly data updates with minimal manual intervention.

This project provides a practical approach to implementing MLops and serves as a template for scalable, production-ready machine learning workflows.
