
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Project paths
ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Load dataset
housing = fetch_california_housing(as_frame=True)

df = housing.frame


# Separate features and target
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Define models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "Lasso": make_pipeline(
        StandardScaler(),
        Lasso(alpha=0.01, max_iter=10000)
    )
}


# Train and evaluate models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# Create results table
results_df = pd.DataFrame(results)

print(results_df.round(4))


# Save results
results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)
