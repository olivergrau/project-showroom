from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from starter.src.ml.data import process_data
from sklearn.metrics import precision_score, recall_score, fbeta_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier and returns the model.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained Random Forest model.
    """
    # Define the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Optional: Perform hyperparameter tuning with GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="f1",  # Optimize for F1 score
        verbose=1,
        n_jobs=-1,
    )

    # Fit the model with GridSearchCV
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)

    # Return the best model
    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Use the model's predict method to generate predictions
    preds = model.predict(X)
    return preds


def test_slices(data, model, encoder, lb, categorical_features, label_col):
    """
    Tests model performance on slices of the data based on categorical features.

    Inputs
    ------
    data : pd.DataFrame
        The dataset to evaluate.
    model : sklearn model
        The trained machine learning model.
    encoder : sklearn.preprocessing.OneHotEncoder
        The fitted encoder for categorical features.
    lb : sklearn.preprocessing.LabelBinarizer
        The fitted label binarizer for the target label.
    categorical_features : list[str]
        List of categorical features for slicing.
    label_col : str
        The name of the target label column.

    Returns
    -------
    slice_metrics : dict
        A dictionary where each key is a feature-value pair and each value is a tuple of precision, recall, and F1 score
    """

    slice_metrics = {}

    # Loop through each categorical feature
    for feature in categorical_features:
        print(f"Evaluating slices for feature: {feature}")

        # Get unique values of the feature
        unique_values = data[feature].unique()

        # Evaluate model performance for each value
        for value in unique_values:
            # Create the slice
            slice_data = data[data[feature] == value]

            # Process the slice data
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=categorical_features,
                label=label_col,
                training=False,
                encoder=encoder,
                lb=lb
            )

            # Predict on the slice
            preds = model.predict(X_slice)

            # Calculate metrics
            precision = precision_score(y_slice, preds, zero_division=1)
            recall = recall_score(y_slice, preds, zero_division=1)
            fbeta = fbeta_score(y_slice, preds, beta=1, zero_division=1)

            # Store the metrics
            slice_metrics[f"{feature}={value}"] = (precision, recall, fbeta)

    return slice_metrics
