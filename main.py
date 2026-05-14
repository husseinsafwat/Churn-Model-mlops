from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

app = FastAPI(title="Churn Prediction API")


class CustomerFeatures(BaseModel):
    CreditScore:       int
    Geography:         str
    Gender:            str
    Age:               int
    Tenure:            int
    Balance:           float
    NumOfProducts:     int
    HasCrCard:         int
    IsActiveMember:    int
    EstimatedSalary:   float


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    result = predict(customer.model_dump())
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)