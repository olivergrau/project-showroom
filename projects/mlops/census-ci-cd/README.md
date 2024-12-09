# Machine Learning Model Deployment with FastAPI and Heroku

## Project Overview

This project demonstrates the end-to-end process of developing, testing, and deploying a machine learning model using FastAPI for API creation and Heroku for deployment. The model is built to analyze census data, and the deployed API allows users to interact with the model for predictions. The workflow incorporates best practices in version control, continuous integration (CI), and unit testing, ensuring a robust and production-ready solution.

---

## Key Features

### 1. **Data Preparation**
   - Process raw census data to clean and prepare it for training.
   - Handle messy data with cleaning steps, such as removing unnecessary spaces.

### 2. **Machine Learning Model**
   - Develop a machine learning model using the provided starter code.
   - Train the model on clean data and save the trained model for inference.
   - Evaluate the model's performance, including generating performance metrics on data slices (e.g., categorical features).
   - Document the model's capabilities and limitations in a model card.

### 3. **API Development**
   - Build a RESTful API using FastAPI, including:
     - A `GET` endpoint for a welcome message.
     - A `POST` endpoint for model inference.
     - Type hinting and Pydantic models to define and validate the API request structure.
   - Ensure compatibility with dataset feature names containing special characters like hyphens.

### 4. **Testing**
   - Write unit tests for core model functions and API endpoints.
   - Include tests for the `GET` endpoint and multiple `POST` requests to validate predictions.

### 5. **Deployment**
   - Deploy the API to Heroku with CI/CD integration using GitHub Actions.
   - Configure automatic deployments triggered by successful CI/CD runs.
   - Address deployment-specific concerns like path differences between local and production environments.

### 6. **Live API Interaction**
   - Create a Python script to send `POST` requests to the live API, demonstrating its functionality.

---

## Applications Demonstrated

- **Model Development**: Train, evaluate, and save machine learning models for production use.
- **API Integration**: Build a robust RESTful API for seamless interaction with the model.
- **Deployment and CI/CD**: Automate deployment and ensure code quality using GitHub Actions and Heroku.
- **Data Engineering**: Clean and preprocess messy datasets for machine learning workflows.

This project provides a comprehensive example of building, deploying, and testing machine learning systems in real-world scenarios, showcasing the importance of automation, testing, and scalable deployment.
