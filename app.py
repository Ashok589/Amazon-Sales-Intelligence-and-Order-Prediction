from pathlib import Path
import pickle
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "amazon_sales_50000_ml_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "amazon_demand_random_forest.joblib"
PREDICTIONS_PATH = BASE_DIR / "reports" / "test_predictions.csv"
REQUIRED_COLUMNS = {
    "Date",
    "Order_ID",
    "Product",
    "Category",
    "Quantity",
    "Total_Sales",
    "Order_Status",
}
EXPORT_COLUMNS = [
    "Date",
    "Order_ID",
    "Product",
    "Category",
    "Quantity",
    "Total_Sales",
    "Order_Status",
]
MODEL_FEATURES = [
    "Day_Of_Week",
    "Day_Of_Month",
    "Month",
    "Is_Weekend",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_Mean_7",
    "Product",
]
PAGES = [
    "Executive overview",
    "Sales analytics",
    "Product intelligence",
    "Demand forecast",
    "Model evaluation",
    "Data quality",
]

st.set_page_config(
    page_title="Amazon Sales Intelligence and Order Prediction",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl="15m", max_entries=1, show_spinner="Loading sales data...")
def load_dataset(
    file_path: str, modified_time: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(file_path)
    source.columns = source.columns.astype(str).str.strip()

    missing_columns = REQUIRED_COLUMNS - set(source.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    quality = source.copy()
    quality["Date"] = pd.to_datetime(quality["Date"], errors="coerce")
    for column in ("Quantity", "Total_Sales"):
        quality[column] = pd.to_numeric(quality[column], errors="coerce")

    data = quality.dropna(
        subset=["Date", "Product", "Quantity", "Total_Sales"]
    ).copy()
    data["Product"] = data["Product"].astype("string").str.strip()
    data = data[data["Product"].notna() & data["Product"].ne("")].copy()

    for column, fallback in (
        ("Category", "Uncategorized"),
        ("Order_Status", "Unknown"),
    ):
        data[column] = (
            data[column].astype("string").str.strip().replace("", pd.NA).fillna(fallback)
        )

    return source, quality, data


@st.cache_resource(max_entries=1)
def load_model(
    file_path: str, modified_time: float
) -> tuple[Any, dict[str, Any]]:
    artifact = joblib.load(file_path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        metadata = artifact.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Model artifact metadata must be a dictionary.")
        if metadata.get("features") != MODEL_FEATURES:
            raise ValueError("Model artifact feature schema is missing or incompatible.")
    else:
        model = artifact
        metadata = {}
    if not callable(getattr(model, "predict", None)):
        raise TypeError("The model artifact does not provide a predict method.")
    return model, metadata


@st.cache_data(ttl="15m", max_entries=1)
def load_test_predictions(file_path: str, modified_time: float) -> pd.DataFrame:
    predictions = pd.read_csv(file_path, parse_dates=["Date"])
    required = {
        "Date",
        "Product",
        "Units_Sold",
        "Seasonal_Naive_Predicted",
        "Model_Predicted",
        "Absolute_Error",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Test predictions are missing required columns: {sorted(missing)}"
        )
    return predictions


if not DATA_PATH.is_file():
    st.error("The sales dataset is not available at the configured project path.")
    st.code("data/amazon_sales_50000_ml_dataset.csv", language="text")
    st.stop()

try:
    source_df, quality_df, df = load_dataset(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
    st.error("The sales dataset could not be loaded or validated.")
    st.code(f"{type(exc).__name__}: {exc}", language="text")
    st.stop()

if df.empty:
    st.error("The dataset contains no usable sales records.")
    st.stop()

model = None
model_metadata: dict[str, Any] = {}
model_error = None
if MODEL_PATH.is_file():
    try:
        model, model_metadata = load_model(
            str(MODEL_PATH), MODEL_PATH.stat().st_mtime
        )
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        ImportError,
    ) as exc:
        model_error = exc

minimum_date = df["Date"].min().date()
maximum_date = df["Date"].max().date()
categories = sorted(df["Category"].dropna().unique().tolist())
statuses = sorted(df["Order_Status"].dropna().unique().tolist())
products = sorted(df["Product"].dropna().unique().tolist())

with st.sidebar:
    st.title("Amazon Sales Intelligence and Order Prediction")
    st.caption("Analytics workspace · sample data")
    st.divider()

    page = st.radio("Workspace", PAGES, label_visibility="visible")

    st.divider()
    st.subheader("Report filters")
    selected_categories = st.multiselect(
        "Categories", categories, default=categories
    )
    selected_statuses = st.multiselect(
        "Order status", statuses, default=statuses
    )
    selected_products = st.multiselect(
        "Products", products, default=products
    )
    selected_dates = st.date_input(
        "Reporting period",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = minimum_date, maximum_date

    st.divider()
    st.subheader("System status")
    st.success(f"Dataset ready · {len(df):,} usable records")
    if model is not None:
        st.success(
            f"Demand model ready · {model_metadata.get('model_version', 'loaded')}"
        )
    elif model_error is not None:
        st.warning("Demand model could not be loaded")
    else:
        st.warning("Demand model not installed")

    with st.expander("Technical diagnostics"):
        st.caption("Project directory")
        st.code(str(BASE_DIR), language="text")
        st.caption("Dataset")
        st.code(str(DATA_PATH), language="text")
        st.caption("Expected model")
        st.code(str(MODEL_PATH), language="text")
        if model_metadata.get("test_metrics"):
            model_test_metrics = model_metadata["test_metrics"][-1]
            st.caption("Latest held-out WAPE")
            st.write(
                f"{model_test_metrics.get('wape_percent', float('nan')):.2f}%"
            )
        if model_error is not None:
            st.error(f"{type(model_error).__name__}: {model_error}")

filtered_df = df.loc[
    df["Category"].isin(selected_categories)
    & df["Order_Status"].isin(selected_statuses)
    & df["Product"].isin(selected_products)
    & df["Date"].dt.date.between(start_date, end_date)
].copy()

st.title(page)
st.caption(
    f"Reporting period: {start_date:%d %b %Y} – {end_date:%d %b %Y} "
    f"· {len(filtered_df):,} matching records "
    f"· Data through {maximum_date:%d %b %Y}"
)
st.info(
    "Demonstration using a sample dataset. This is not an official Amazon "
    "application or live operational data."
)


def require_filtered_records() -> bool:
    if filtered_df.empty:
        st.warning("No records match the selected filters. Update the sidebar filters.")
        return False
    return True


def render_download(frame: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


if page == "Executive overview":
    if require_filtered_records():
        sales = float(filtered_df["Total_Sales"].sum())
        units = float(filtered_df["Quantity"].sum())
        orders = int(filtered_df["Order_ID"].nunique())
        cancelled = int(
            filtered_df.loc[
                filtered_df["Order_Status"].eq("Cancelled"), "Order_ID"
            ].nunique()
        )
        cancellation_rate = cancelled / orders if orders else 0.0

        with st.container(horizontal=True):
            with st.container(border=True):
                st.metric("Recorded sales", f"{sales:,.2f}")
            with st.container(border=True):
                st.metric("Units ordered", f"{units:,.0f}")
            with st.container(border=True):
                st.metric("Unique orders", f"{orders:,}")
            with st.container(border=True):
                st.metric("Cancellation rate", f"{cancellation_rate:.1%}")

        st.caption(
            "Recorded sales include every selected order status and should not "
            "be interpreted as realized revenue."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Sales by category")
            category_sales = (
                filtered_df.groupby("Category", as_index=False)["Total_Sales"]
                .sum()
                .rename(columns={"Total_Sales": "Recorded sales"})
                .sort_values("Recorded sales")
            )
            st.bar_chart(
                category_sales,
                x="Category",
                y="Recorded sales",
                horizontal=True,
                alt="Recorded sales by product category",
            )

        with right:
            st.subheader("Monthly sales trend")
            monthly_sales = (
                filtered_df.assign(
                    Month=filtered_df["Date"].dt.to_period("M").astype(str)
                )
                .groupby("Month", as_index=False)["Total_Sales"]
                .sum()
                .rename(columns={"Total_Sales": "Recorded sales"})
            )
            st.line_chart(
                monthly_sales,
                x="Month",
                y="Recorded sales",
                alt="Monthly recorded sales over the reporting period",
            )

        st.subheader("Order status")
        status_summary = (
            filtered_df.groupby("Order_Status")["Order_ID"]
            .nunique()
            .rename("Unique orders")
            .reset_index()
            .sort_values("Unique orders", ascending=False)
        )
        st.dataframe(
            status_summary,
            hide_index=True,
            alt="Unique order counts by order status",
        )

elif page == "Sales analytics":
    if require_filtered_records():
        st.subheader("Sales performance")
        left, right = st.columns(2)
        with left:
            category_sales = (
                filtered_df.groupby("Category", as_index=False)["Total_Sales"]
                .sum()
                .rename(columns={"Total_Sales": "Recorded sales"})
                .sort_values("Recorded sales")
            )
            st.bar_chart(
                category_sales,
                x="Category",
                y="Recorded sales",
                horizontal=True,
                alt="Recorded sales by category",
            )
        with right:
            daily_sales = (
                filtered_df.assign(
                    Sales_Date=filtered_df["Date"].dt.normalize()
                )
                .groupby("Sales_Date", as_index=False)["Total_Sales"]
                .sum()
                .rename(
                    columns={
                        "Sales_Date": "Date",
                        "Total_Sales": "Recorded sales",
                    }
                )
            )
            st.line_chart(
                daily_sales,
                x="Date",
                y="Recorded sales",
                alt="Daily recorded sales over the reporting period",
            )
        st.subheader("Filtered order records")
        safe_export = filtered_df.loc[:, EXPORT_COLUMNS].sort_values("Date")
        st.dataframe(
            safe_export,
            hide_index=True,
            alt="Filtered sales order records without customer personal details",
        )
        render_download(
            safe_export, "filtered_sales_records.csv", "Download filtered records"
        )

elif page == "Product intelligence":
    if require_filtered_records():
        product_limit = st.slider(
            "Products to display", min_value=1, max_value=max(1, len(products)),
            value=min(10, len(products)),
        )
        product_summary = (
            filtered_df.groupby(["Product", "Category"], as_index=False)
            .agg(
                Recorded_Sales=("Total_Sales", "sum"),
                Units_Ordered=("Quantity", "sum"),
                Unique_Orders=("Order_ID", "nunique"),
            )
            .sort_values("Recorded_Sales", ascending=False)
            .head(product_limit)
        )
        chart_data = product_summary.rename(
            columns={
                "Product": "Product",
                "Recorded_Sales": "Recorded sales",
            }
        ).sort_values("Recorded sales")
        st.bar_chart(
            chart_data,
            x="Product",
            y="Recorded sales",
            horizontal=True,
            alt="Top products ranked by recorded sales",
        )
        st.dataframe(
            product_summary,
            hide_index=True,
            column_config={
                "Recorded_Sales": st.column_config.NumberColumn(
                    "Recorded sales", format="%.2f"
                ),
                "Units_Ordered": st.column_config.NumberColumn(
                    "Units ordered", format="%.0f"
                ),
            },
            alt="Product sales, ordered units, and unique order counts",
        )
        render_download(
            product_summary, "product_performance.csv", "Download product report"
        )

elif page == "Demand forecast":
    st.subheader("Next-day demand estimate")
    st.write(
        "Forecasts require a trained, compatible model artifact. Recent daily "
        "order quantities are prefilled from the dataset and can be adjusted."
    )

    if model is None:
        st.warning(
            "Forecasting is disabled because the expected trained model is "
            "unavailable. Analytics and data-quality pages remain available."
        )
        st.code("models/amazon_demand_random_forest.joblib", language="text")
        st.caption(
            "Save a fitted estimator or pipeline with joblib.dump at the path "
            "above. A Python source file renamed with a .joblib extension is "
            "not a trained model artifact."
        )
        if model_error is not None:
            st.error(f"Model load error: {type(model_error).__name__}: {model_error}")
    else:
        forecast_product = st.selectbox("Product", products)
        forecast_date = maximum_date + pd.Timedelta(days=1)
        forecast_timestamp = pd.Timestamp(forecast_date)
        st.caption(
            f"Forecast horizon is fixed to the next day after available data: "
            f"{forecast_date:%d %b %Y}."
        )
        history_dates = pd.date_range(
            end=forecast_timestamp - pd.Timedelta(days=1), periods=14, freq="D"
        )
        product_history = (
            df.loc[
                df["Product"].eq(forecast_product)
                & df["Date"].lt(forecast_timestamp)
                & df["Order_Status"].isin(["Delivered", "Pending"])
            ]
            .assign(Demand_Date=lambda frame: frame["Date"].dt.normalize())
            .groupby("Demand_Date")["Quantity"]
            .sum()
        )
        default_history = (
            product_history.reindex(history_dates, fill_value=0)
            .round()
            .clip(lower=0)
            .astype(int)
            .tolist()
        )
        st.caption(
            "History is aggregated from Delivered and Pending orders; "
            "Cancelled orders are excluded to match model training."
        )

        with st.form("demand_forecast_form"):
            st.caption("Daily history · oldest first")
            history = []
            columns = st.columns(2)
            for index, default_value in enumerate(default_history):
                with columns[index % 2]:
                    history.append(
                        st.number_input(
                            f"Day {index + 1} ({history_dates[index]:%d %b})",
                            min_value=0,
                            max_value=100000,
                            value=int(default_value),
                            step=1,
                            key=f"history_{forecast_product}_{index}",
                        )
                    )
            submitted = st.form_submit_button(
                "Generate demand estimate",
                type="primary",
                icon=":material/online_prediction:",
            )

        if submitted:
            features = pd.DataFrame(
                [
                    {
                        "Day_Of_Week": forecast_timestamp.dayofweek,
                        "Day_Of_Month": forecast_timestamp.day,
                        "Month": forecast_timestamp.month,
                        "Is_Weekend": int(forecast_timestamp.dayofweek >= 5),
                        "Lag_1": history[-1],
                        "Lag_7": history[-7],
                        "Lag_14": history[-14],
                        "Rolling_Mean_7": float(np.mean(history[-7:])),
                        "Product": forecast_product,
                    }
                ]
            )
            try:
                prediction_values = np.asarray(model.predict(features)).reshape(-1)
                if len(prediction_values) != 1 or not np.isfinite(prediction_values[0]):
                    raise ValueError("The model returned an invalid prediction.")
                prediction = float(prediction_values[0])
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                IndexError,
            ) as exc:
                st.error("The model could not generate a forecast for these inputs.")
                st.code(f"{type(exc).__name__}: {exc}", language="text")
            else:
                prediction = max(0.0, prediction)
                rounded_prediction = int(np.ceil(prediction))
                first, second = st.columns(2)
                first.metric("Predicted units", f"{prediction:.1f}")
                second.metric("Rounded planning estimate", f"{rounded_prediction:,}")
                st.caption(
                    "Experimental estimate only. Validate model performance and "
                    "business assumptions before operational use. This point "
                    "estimate has no calibrated prediction interval."
                )
                forecast_result = pd.DataFrame(
                    [
                        {
                            "Product": forecast_product,
                            "Forecast_Date": forecast_date.isoformat(),
                            "Predicted_Units": prediction,
                            "Rounded_Planning_Estimate": rounded_prediction,
                        }
                    ]
                )
                render_download(
                    forecast_result, "demand_forecast.csv", "Download forecast"
                )

elif page == "Model evaluation":
    st.subheader("Model card and holdout performance")
    if not model_metadata:
        st.warning(
            "No model evaluation metadata is available. Train the model with "
            "`python train_model.py` to generate the model card and reports."
        )
    else:
        summary = st.columns(4)
        summary[0].metric("Model version", model_metadata.get("model_version", "—"))
        summary[1].metric("Selected model", model_metadata.get("model_name", "—"))
        summary[2].metric(
            "Test window",
            f"{model_metadata['split'].get('test_start', '—')} to "
            f"{model_metadata['split'].get('test_end', '—')}",
        )
        summary[3].metric(
            "Forecast horizon", model_metadata.get("forecast_horizon", "—")
        )
        st.write(f"**Target:** {model_metadata.get('target', 'Not specified')}")
        st.write(
            f"**Evaluation protocol:** "
            f"{model_metadata.get('evaluation_method', 'Not specified')}"
        )
        row_counts = model_metadata.get("row_counts", {})
        count_columns = st.columns(3)
        count_columns[0].metric(
            "Training product-days", f"{row_counts.get('training', 0):,}"
        )
        count_columns[1].metric(
            "Validation product-days", f"{row_counts.get('validation', 0):,}"
        )
        count_columns[2].metric(
            "Test product-days", f"{row_counts.get('test', 0):,}"
        )

        st.markdown("#### Validation comparison")
        validation_results = pd.DataFrame(
            model_metadata.get("validation_metrics", [])
        )
        if not validation_results.empty:
            st.dataframe(
                validation_results,
                hide_index=True,
                column_config={
                    "mae": st.column_config.NumberColumn("MAE (units)", format="%.3f"),
                    "rmse": st.column_config.NumberColumn("RMSE (units)", format="%.3f"),
                    "wape_percent": st.column_config.NumberColumn("WAPE", format="%.2f%%"),
                    "mean_error": st.column_config.NumberColumn(
                        "Mean error (units)", format="%.3f"
                    ),
                },
                alt="Validation metrics for the seasonal baseline and candidate models",
            )

        st.markdown("#### Untouched test-set comparison")
        test_results = pd.DataFrame(model_metadata.get("test_metrics", []))
        if not test_results.empty:
            st.dataframe(
                test_results,
                hide_index=True,
                column_config={
                    "mae": st.column_config.NumberColumn("MAE (units)", format="%.3f"),
                    "rmse": st.column_config.NumberColumn("RMSE (units)", format="%.3f"),
                    "wape_percent": st.column_config.NumberColumn("WAPE", format="%.2f%%"),
                    "mean_error": st.column_config.NumberColumn(
                        "Mean error (units)", format="%.3f"
                    ),
                },
                alt="Final holdout metrics for the seasonal baseline and selected model",
            )

        if PREDICTIONS_PATH.is_file():
            try:
                test_predictions = load_test_predictions(
                    str(PREDICTIONS_PATH), PREDICTIONS_PATH.stat().st_mtime
                )
            except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                st.error("The saved test-prediction report could not be loaded.")
                st.code(f"{type(exc).__name__}: {exc}", language="text")
            else:
                st.markdown("#### Test-period forecast review")
                test_products = sorted(test_predictions["Product"].unique().tolist())
                inspected_product = st.selectbox(
                    "Product to inspect", test_products
                )
                product_predictions = (
                    test_predictions.loc[
                        test_predictions["Product"].eq(inspected_product),
                        [
                            "Date",
                            "Units_Sold",
                            "Seasonal_Naive_Predicted",
                            "Model_Predicted",
                        ],
                    ]
                    .rename(
                        columns={
                            "Units_Sold": "Actual units",
                            "Seasonal_Naive_Predicted": "Seasonal baseline",
                            "Model_Predicted": "Model forecast",
                        }
                    )
                    .sort_values("Date")
                )
                st.line_chart(
                    product_predictions,
                    x="Date",
                    y=["Actual units", "Seasonal baseline", "Model forecast"],
                    alt=f"Actual and predicted daily demand for {inspected_product}",
                )
                product_error = (
                    test_predictions.groupby("Product", as_index=False)
                    .agg(
                        Actual_Units=("Units_Sold", "sum"),
                        Absolute_Error=("Absolute_Error", "sum"),
                        MAE=("Absolute_Error", "mean"),
                    )
                )
                product_error["WAPE_Percent"] = np.where(
                    product_error["Actual_Units"].ne(0),
                    product_error["Absolute_Error"]
                    / product_error["Actual_Units"]
                    * 100,
                    np.nan,
                )
                st.dataframe(
                    product_error.sort_values("WAPE_Percent", ascending=False),
                    hide_index=True,
                    column_config={
                        "MAE": st.column_config.NumberColumn(
                            "MAE (units)", format="%.3f"
                        ),
                        "WAPE_Percent": st.column_config.NumberColumn(
                            "WAPE", format="%.2f%%"
                        ),
                    },
                    alt="Holdout forecast error by product",
                )
        else:
            st.info(
                "Test-period predictions are not present. Run `python train_model.py` "
                "to generate the evaluation report."
            )

elif page == "Data quality":
    st.subheader("Dataset health")
    missing_cells = int(quality_df.isna().sum().sum())
    duplicate_rows = int(source_df.duplicated().sum())
    invalid_rows = len(source_df) - len(df)

    first, second, third, fourth = st.columns(4)
    first.metric("Source rows", f"{len(source_df):,}")
    second.metric("Usable rows", f"{len(df):,}")
    third.metric("Rows excluded", f"{invalid_rows:,}")
    fourth.metric("Duplicate rows", f"{duplicate_rows:,}")
    st.metric("Missing or invalid cells", f"{missing_cells:,}")

    missing_report = pd.DataFrame(
        {
            "Column": quality_df.columns,
            "Missing_or_Invalid": [
                int(quality_df[column].isna().sum()) for column in quality_df
            ],
            "Percentage": [
                float(quality_df[column].isna().mean() * 100)
                for column in quality_df
            ],
        }
    ).sort_values("Percentage", ascending=False)
    st.subheader("Completeness by column")
    st.dataframe(
        missing_report,
        hide_index=True,
        column_config={
            "Percentage": st.column_config.NumberColumn(
                "Missing or invalid (%)", format="%.2f%%"
            )
        },
        alt="Missing and invalid value counts by source column",
    )
    st.subheader("Source order-status distribution")
    source_status = (
        source_df["Order_Status"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Order status")
        .reset_index(name="Rows")
    )
    st.bar_chart(
        source_status,
        x="Order status",
        y="Rows",
        alt="Source records by order status",
    )
    render_download(
        missing_report, "data_quality_report.csv", "Download quality report"
    )

st.divider()
st.caption(
    "Amazon Sales Intelligence and Order Prediction demo · sample data · no customer-level personal "
    "information is included in report exports."
)
