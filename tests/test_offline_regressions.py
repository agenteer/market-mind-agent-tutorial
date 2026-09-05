import os
import unittest
from unittest.mock import patch

from src.common.config import check_api_key
from src.common.tools_yf import get_financial_metrics


class FakeTicker:
    info = {
        "shortName": "Example Inc.",
        "marketCap": 1_000_000_000,
        "trailingPE": 12.5,
        "dividendYield": 0.33,
        "fiftyTwoWeekHigh": 20.0,
        "fiftyTwoWeekLow": 10.0,
        "totalRevenue": 2_000_000_000,
        "profitMargins": 0.1,
        "returnOnEquity": 0.2,
    }


class OfflineRegressionTests(unittest.TestCase):
    def test_placeholder_api_key_has_an_actionable_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "your_api_key_here"}, clear=True):
            with self.assertRaisesRegex(ValueError, "replace the placeholder"):
                check_api_key()

    @patch("src.common.tools_yf.yf.Ticker", return_value=FakeTicker())
    def test_dividend_yield_is_not_multiplied_twice(self, _ticker):
        result = get_financial_metrics("EXM")
        self.assertIn("Dividend Yield: 0.33%", result)


if __name__ == "__main__":
    unittest.main()
