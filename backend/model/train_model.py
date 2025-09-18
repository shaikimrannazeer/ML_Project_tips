import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # For saving the model

def train_model(df):

    """Train a machine learning model on the provided DataFrame."""

    # Separate tips and rest of the variables as X and y
    X = df.drop("tip", axis=1)
    y = df["tip"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["category"]).columns
    numerical_cols = X.select_dtypes(exclude=["category"]).columns

    # One hot encoding for categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(), categorical_cols),
        ]
    )
    # create a pipeline that first transforms the data then fits the model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    return model
