from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "amazon_sales_50000_ml_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "amazon_demand_random_forest.joblib"
DEFAULT_METRICS_PATH = PROJECT_DIR / "reports" / "model_metrics.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_DIR / "reports" / "test_predictions.csv"
REQUIRED_COLUMNS = {"Date", "Order_ID", "Product", "Quantity", "Order_Status"}
NUMERIC_FEATURES = [
    "Day_Of_Week",
    "Day_Of_Month",
    "Month",
    "Is_Weekend",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_Mean_7",
]
CATEGORICAL_FEATURES = ["Product"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Units_Sold"


def load_orders(path: Path) -> tuple[pd.DataFrame, int]:
    orders = pd.read_csv(path)
    orders.columns = orders.columns.astype(str).str.strip()
    missing = REQUIRED_COLUMNS - set(orders.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    orders["Date"] = pd.to_datetime(orders["Date"], errors="coerce").dt.normalize()
    orders["Quantity"] = pd.to_numeric(orders["Quantity"], errors="coerce")
    orders["Product"] = orders["Product"].astype("string").str.strip()
    orders["Order_Status"] = (
        orders["Order_Status"].astype("string").str.strip().str.title()
    )
    valid = (
        orders["Date"].notna()
        & orders["Quantity"].notna()
        & orders["Quantity"].ge(0)
        & orders["Product"].notna()
        & orders["Product"].ne("")
        & orders["Order_Status"].notna()
    )
    invalid_rows = int((~valid).sum())
    return orders.loc[valid].copy(), invalid_rows


def build_daily_demand(orders: pd.DataFrame) -> pd.DataFrame:
    demand_orders = orders.loc[
        orders["Order_Status"].isin(["Delivered", "Pending"])
    ]
    if demand_orders.empty:
        raise ValueError("No Delivered or Pending orders are available for training.")

    return (
        demand_orders.groupby(["Date", "Product"], as_index=False)
        .agg(Units_Sold=("Quantity", "sum"))
        .sort_values(["Product", "Date"])
        .reset_index(drop=True)
    )


def build_supervised_data(daily_demand: pd.DataFrame) -> pd.DataFrame:
    if daily_demand.empty:
        raise ValueError("Daily demand data is empty.")

    dates = pd.date_range(
        daily_demand["Date"].min().normalize(),
        daily_demand["Date"].max().normalize(),
        freq="D",
    )
    product_names = sorted(daily_demand["Product"].dropna().unique())
    complete_index = pd.MultiIndex.from_product(
        [product_names, dates], names=["Product", "Date"]
    )
    complete = (
        daily_demand.assign(Date=daily_demand["Date"].dt.normalize())
        .set_index(["Product", "Date"])
        .reindex(complete_index)
        .reset_index()
    )
    complete[TARGET] = complete[TARGET].fillna(0.0).astype(float)
    complete = complete.sort_values(["Product", "Date"]).reset_index(drop=True)

    grouped_demand = complete.groupby("Product", sort=False)[TARGET]
    complete["Day_Of_Week"] = complete["Date"].dt.dayofweek
    complete["Day_Of_Month"] = complete["Date"].dt.day
    complete["Month"] = complete["Date"].dt.month
    complete["Is_Weekend"] = complete["Day_Of_Week"].ge(5).astype(int)
    complete["Lag_1"] = grouped_demand.shift(1)
    complete["Lag_7"] = grouped_demand.shift(7)
    complete["Lag_14"] = grouped_demand.shift(14)
    complete["Rolling_Mean_7"] = grouped_demand.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).mean()
    )

    supervised = complete.dropna(subset=FEATURES).copy()
    return supervised.sort_values(["Date", "Product"]).reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame, validation_days: int = 30, test_days: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if validation_days < 1 or test_days < 1:
        raise ValueError("Validation and test windows must each be at least one day.")

    last_date = data["Date"].max().normalize()
    test_start = last_date - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    train = data.loc[data["Date"] < validation_start].copy()
    validation = data.loc[
        data["Date"].ge(validation_start) & data["Date"].lt(test_start)
    ].copy()
    test = data.loc[data["Date"].ge(test_start)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "The dataset must cover more days than the combined validation and "
            "test windows, plus the 14-day lag warm-up period."
        )

    boundaries = {
        "train_start": train["Date"].min().date().isoformat(),
        "train_end": train["Date"].max().date().isoformat(),
        "validation_start": validation["Date"].min().date().isoformat(),
        "validation_end": validation["Date"].max().date().isoformat(),
        "test_start": test["Date"].min().date().isoformat(),
        "test_end": test["Date"].max().date().isoformat(),
    }
    return train, validation, test, boundaries


def make_pipeline(min_samples_leaf: int, max_depth: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_metrics(
    actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray
) -> dict[str, float | None]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    errors = actual_values - predicted_values
    denominator = float(np.abs(actual_values).sum())
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "wape_percent": (
            float(np.abs(errors).sum() / denominator * 100)
            if denominator > 0
            else None
        ),
        "mean_error": float(errors.mean()),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def train_and_evaluate(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    validation_days: int = 30,
    test_days: int = 30,
) -> dict[str, Any]:
    orders, invalid_rows = load_orders(data_path)
    daily_demand = build_daily_demand(orders)
    supervised = build_supervised_data(daily_demand)
    train, validation, test, boundaries = chronological_split(
        supervised, validation_days=validation_days, test_days=test_days
    )

    target_validation = validation[TARGET]
    validation_results: list[dict[str, Any]] = []
    baseline_validation = calculate_metrics(
        target_validation, validation["Lag_7"]
    )
    validation_results.append(
        {"model": "Seasonal naive (Lag 7)", **baseline_validation}
    )

    candidates = [
        ("Random forest (regularized)", 2, 15),
        ("Random forest (shallow)", 4, 10),
    ]
    best_pipeline: Pipeline | None = None
    best_name = ""
    best_leaf = 0
    best_depth = 0
    best_validation_mae = float("inf")

    for name, min_samples_leaf, max_depth in candidates:
        pipeline = make_pipeline(min_samples_leaf, max_depth)
        pipeline.fit(train[FEATURES], train[TARGET])
        predicted = np.maximum(pipeline.predict(validation[FEATURES]), 0.0)
        metrics = calculate_metrics(target_validation, predicted)
        validation_results.append({"model": name, **metrics})
        if metrics["mae"] is not None and metrics["mae"] < best_validation_mae:
            best_pipeline = pipeline
            best_name = name
            best_leaf = min_samples_leaf
            best_depth = max_depth
            best_validation_mae = metrics["mae"]

    if best_pipeline is None:
        raise RuntimeError("No candidate model produced valid validation metrics.")

    # The final test window is used only once for an unbiased holdout estimate.
    test_predictions = np.maximum(best_pipeline.predict(test[FEATURES]), 0.0)
    baseline_test_predictions = test["Lag_7"].to_numpy(dtype=float)
    test_results = [
        {
            "model": "Seasonal naive (Lag 7)",
            **calculate_metrics(test[TARGET], baseline_test_predictions),
        },
        {
            "model": best_name,
            **calculate_metrics(test[TARGET], test_predictions),
        },
    ]
    prediction_report = test[["Date", "Product", TARGET]].copy()
    prediction_report["Seasonal_Naive_Predicted"] = baseline_test_predictions
    prediction_report["Model_Predicted"] = test_predictions
    prediction_report["Absolute_Error"] = np.abs(
        prediction_report[TARGET] - test_predictions
    )

    final_pipeline = make_pipeline(best_leaf, best_depth)
    pre_test = supervised.loc[supervised["Date"] < test["Date"].min()]
    final_pipeline.fit(pre_test[FEATURES], pre_test[TARGET])

    metadata: dict[str, Any] = {
        "model_version": "1.0.0",
        "model_name": best_name,
        "target": "daily product units ordered (Delivered and Pending; Cancelled excluded)",
        "features": FEATURES,
        "validation_metrics": validation_results,
        "test_metrics": test_results,
        "split": boundaries,
        "row_counts": {
            "input_orders": int(len(orders) + invalid_rows),
            "invalid_rows_excluded": invalid_rows,
            "training": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "products": int(daily_demand["Product"].nunique()),
        },
        "forecast_horizon": "one day ahead, using observed prior demand",
        "evaluation_method": (
            "Chronological holdout. The validation window selects the "
            "Random Forest configuration; test data is not used for selection."
        ),
    }

    artifact = {"model": final_pipeline, "metadata": metadata}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_name(f"{model_path.name}.tmp")
    joblib.dump(artifact, temporary_model_path, compress=3)
    temporary_model_path.replace(model_path)

    atomic_write_json(metrics_path, metadata)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_predictions_path = predictions_path.with_name(
        f"{predictions_path.name}.tmp"
    )
    prediction_report.to_csv(temporary_predictions_path, index=False)
    temporary_predictions_path.replace(predictions_path)

    print(f"Model: {best_name}")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")
    print("\nValidation metrics:")
    print(pd.DataFrame(validation_results).to_string(index=False))
    print("\nUntouched test metrics:")
    print(pd.DataFrame(test_results).to_string(index=False))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the one-day product-demand model."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH
    )
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )


if __name__ == "__main__":
    main()
