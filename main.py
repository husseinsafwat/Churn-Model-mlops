import logging
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PREPROCESSOR_PATH = "preprocessor.pkl"
MODEL_PATH        = "model.pkl"

FEATURE_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    with open(PREPROCESSOR_PATH, "rb") as f:
        app.state.preprocessor = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        app.state.model = pickle.load(f)
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    CreditScore:      int   = Field(..., ge=400, le=800)
    Geography:        str   = Field(..., pattern="^(France|Spain|Germany)$")
    Gender:           str   = Field(..., pattern="^(Male|Female)$")
    Age:              int   = Field(..., ge=20, le=70)
    Tenure:           int   = Field(..., ge=0, le=10)
    Balance:          float = Field(..., ge=0.0, le=100_000.0)
    NumOfProducts:    int   = Field(..., ge=1, le=4)
    HasCrCard:        int   = Field(..., ge=0, le=1)
    IsActiveMember:   int   = Field(..., ge=0, le=1)
    EstimatedSalary:  float = Field(..., ge=20_000.0, le=120_000.0)


@app.get("/")
def home():
    logger.info("Home endpoint called")
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    logger.info("Prediction request received: %s", customer.model_dump())
    X             = pd.DataFrame([customer.model_dump()])[FEATURE_COLUMNS]
    X_transformed = app.state.preprocessor.transform(X)
    X_ready       = pd.DataFrame(X_transformed, columns=app.state.preprocessor.get_feature_names_out())
    prediction    = int(app.state.model.predict(X_ready)[0])
    probability   = float(app.state.model.predict_proba(X_ready)[0][1])
    result        = {"prediction": prediction, "probability": probability}
    logger.info("Prediction result: %s", result)
    return result


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Churn Prediction API on 127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # Finish main