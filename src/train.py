"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    mlflow.sklearn.log_model(col_transf, artifact_path="preprocessor")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train , model , model_name):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    model = model
    model.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(model, model_name , signature=signature )

    ### Log the data
    dataset = mlflow.data.from_pandas(X_train, source="dataset/Churn_Modelling.csv", name="training_data")
    mlflow.log_input(dataset, context="training")

    return model


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")


    ### Set the experiment name
    mlflow.set_experiment("Bank_Churn_Predictionlast")

    models = {'svm' : SVC(C = 100 , gamma = .01) , 'adaboost' :
              AdaBoostClassifier(n_estimators=1000 ) ,
              'randomforest' : RandomForestClassifier(n_estimators=1000 , max_samples=1500)
                                                                         }
    
    ### Start a new run and leave all the main function code as part of the experiment
    for model_name , model1 in models.items() : 
        with mlflow.start_run(run_name = model_name):

            df = pd.read_csv("dataset/Churn_Modelling.csv")
            col_transf, X_train, X_test, y_train, y_test = preprocess(df)

           


            model = train(X_train, y_train , model1 , model_name)
             ### Log the max_iter parameter or saving model params 
            mlflow.log_params(model.get_params())
            y_pred = model.predict(X_test)



            ### Log metrics after calculating them
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)


            ### Log tag
            mlflow.set_tag("version", "1.0.0")


            conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
            conf_mat_disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_mat, display_labels=model.classes_
            )
            conf_mat_disp.plot()

            # Log the image as an artifact in MLflow
            plt.savefig("confusion_matrix_for"+model_name+".png")
            mlflow.log_artifact("confusion_matrix_for"+model_name+".png")
            plt.show()


if __name__ == "__main__":
    main()