from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from starter.src.ml.data import process_data
from starter.src.ml.model import inference

# Load pre-trained model and preprocessing objects
MODEL_PATH = "./starter/model/trained_model.pkl"
ENCODER_PATH = "./starter/model/encoder.pkl"
LB_PATH = "./starter/model/label_binarizer.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
lb = joblib.load(LB_PATH)

# Create FastAPI app
app = FastAPI()

# Pydantic model for request body
class CensusData(BaseModel):
    age: int
    workclass: str = Field(alias="workclass")
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    # Example request
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


@app.get("/")
def read_root():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to the Census Salary Prediction API!"}


@app.post("/predict")
def predict(data: CensusData):
    """Predict the salary class for the given input data."""
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict(by_alias=True)])

        # Process the input data
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        X, _, _, _ = process_data(
            input_df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Perform inference
        prediction = inference(model, X)
        prediction_label = lb.inverse_transform(prediction)[0]

        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
