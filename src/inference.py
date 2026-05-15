import pickle
import pandas as pd

PREPROCESSOR_PATH = "mlartifacts/921095223212119000/0447d40695494f068500d1de4dfd1996/artifacts/preprocessor/model.pkl"
MODEL_PATH        = "mlartifacts/921095223212119000/0447d40695494f068500d1de4dfd1996/artifacts/randomforest/model.pkl"

FEATURE_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


def load_model():
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return preprocessor, model


def predict(preprocessor, model, customer: dict) -> dict:
    X             = pd.DataFrame([customer])[FEATURE_COLUMNS]
    X_transformed = preprocessor.transform(X)
    X_ready       = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
    prediction    = int(model.predict(X_ready)[0])
    probability   = float(model.predict_proba(X_ready)[0][1])
    return {"prediction": prediction, "probability": probability}