import unittest

import numpy as np
import pandas as pd

from train_model import (
    build_daily_demand,
    build_supervised_data,
    calculate_metrics,
    chronological_split,
)


class DemandPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        self.orders = pd.DataFrame(
            {
                "Date": list(dates) + [dates[50]],
                "Product": ["Widget"] * 101,
                "Order_ID": [f"ORD-{index}" for index in range(101)],
                "Quantity": list(range(1, 101)) + [500],
                "Order_Status": ["Delivered"] * 100 + ["Cancelled"],
            }
        )

    def test_cancelled_orders_are_not_demand(self) -> None:
        daily = build_daily_demand(self.orders)
        self.assertEqual(len(daily), 100)
        self.assertEqual(float(daily["Units_Sold"].sum()), sum(range(1, 101)))

    def test_lag_features_only_use_prior_demand(self) -> None:
        daily = build_daily_demand(self.orders)
        features = build_supervised_data(daily)
        row = features.loc[features["Date"].eq(pd.Timestamp("2026-01-15"))].iloc[0]
        self.assertEqual(row["Lag_1"], 14)
        self.assertEqual(row["Lag_7"], 8)
        self.assertEqual(row["Lag_14"], 1)
        self.assertEqual(row["Rolling_Mean_7"], np.mean(range(8, 15)))

    def test_split_is_chronological_and_disjoint(self) -> None:
        features = build_supervised_data(build_daily_demand(self.orders))
        train, validation, test, boundaries = chronological_split(
            features, validation_days=10, test_days=10
        )
        self.assertLess(train["Date"].max(), validation["Date"].min())
        self.assertLess(validation["Date"].max(), test["Date"].min())
        self.assertEqual(boundaries["test_end"], "2026-04-10")

    def test_metrics_handle_zero_demand_without_division_error(self) -> None:
        metrics = calculate_metrics(
            np.array([0.0, 2.0]), np.array([1.0, 3.0])
        )
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["wape_percent"], 100.0)

    def test_wape_is_none_when_test_actuals_are_all_zero(self) -> None:
        metrics = calculate_metrics(np.zeros(2), np.array([2.0, 1.0]))
        self.assertIsNone(metrics["wape_percent"])


if __name__ == "__main__":
    unittest.main()
