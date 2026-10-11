from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_all_pages_render_without_exceptions(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
        for page in (
            "Executive overview",
            "Sales analytics",
            "Product intelligence",
            "Demand forecast",
            "Model evaluation",
            "Data quality",
        ):
            with self.subTest(page=page):
                app.radio[0].set_value(page).run()
                self.assertEqual([], list(app.exception))

    def test_next_day_prediction_submits(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
        app.radio[0].set_value("Demand forecast").run()
        app.button[0].click().run()
        self.assertEqual([], list(app.exception))
        self.assertEqual([], list(app.error))
        self.assertTrue(
            any(metric.label == "Predicted units" for metric in app.metric)
        )

    def test_sales_export_omits_customer_fields(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
        app.radio[0].set_value("Sales analytics").run()
        self.assertEqual([], list(app.exception))
        export_columns = set(app.dataframe[0].value.columns)
        self.assertNotIn("Customer_Name", export_columns)
        self.assertNotIn("Customer_ID", export_columns)

    def test_empty_filter_selection_is_handled(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
        app.radio[0].set_value("Sales analytics").run()
        app.multiselect[0].set_value([]).run()
        self.assertEqual([], list(app.exception))
        self.assertTrue(
            any("No records match" in warning.value for warning in app.warning)
        )


if __name__ == "__main__":
    unittest.main()
