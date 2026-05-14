import mlflow.sklearn
import pandas as pd

TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID       = "0447d40695494f068500d1de4dfd1996"

FEATURE_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

mlflow.set_tracking_uri(TRACKING_URI)
PREPROCESSOR = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/preprocessor")
MODEL        = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/randomforest")

def predict(customer: dict) -> dict:
    X            = pd.DataFrame([customer])[FEATURE_COLUMNS]
    X_transformed = PREPROCESSOR.transform(X)
    X_ready      = pd.DataFrame(X_transformed, columns=PREPROCESSOR.get_feature_names_out())
    prediction   = int(MODEL.predict(X_ready)[0])
    probability  = float(MODEL.predict_proba(X_ready)[0][1])
    return {"prediction": prediction, "probability": probability}
