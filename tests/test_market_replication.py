import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from model.carrmadan import (
    calculate_strike_quadrature_widths,
    format_portfolio_summary,
    run_market_replication_pipeline,
    select_eligible_option_strikes,
)
from model.draw import SmoothPayoffDrawer
from model.loader import (
    EUROPEAN_INDEX_TICKERS,
    create_option_dataframe,
    get_option_expirations,
    get_ticker_market_data,
)


class MarketReplicationTests(unittest.TestCase):
    """Check irregular-strike quadrature and minimum-volume selection."""

    def test_quadrature_widths_use_midpoints_and_full_edge_intervals(self) -> None:
        strikes = np.array([2.0, 5.0, 11.0, 20.0])

        widths = calculate_strike_quadrature_widths(strikes)

        np.testing.assert_allclose(widths, [3.0, 4.5, 7.5, 9.0])

    def test_portfolio_summary_explains_fixed_cash_sign(self) -> None:
        summary = format_portfolio_summary(
            put_count=1,
            call_count=2,
            asset_quantity=0.5,
            fixed_component=-12.0,
            initial_price=100.0,
        )

        self.assertIn(
            "positive = receive, negative = pay",
            summary,
        )
        self.assertIn("-12.000000", summary)

    def test_market_pipeline_filters_by_volume_and_uses_irregular_strikes(self) -> None:
        grid = np.linspace(1.0, 19.0, 181)
        payoff = (grid - 10.0) ** 2
        result = run_market_replication_pipeline(
            strike_grid=grid,
            payoff_values=payoff,
            initial_price=10.0,
            put_strikes=np.array([2.0, 6.0, 9.0]),
            put_volumes=np.array([20.0, 5.0, 20.0]),
            call_strikes=np.array([11.0, 15.0, 18.0]),
            call_volumes=np.array([20.0, 20.0, 5.0]),
            minimum_volume=10.0,
        )

        np.testing.assert_array_equal(result.put_strikes, [2.0, 9.0])
        np.testing.assert_array_equal(result.call_strikes, [11.0, 15.0])
        np.testing.assert_allclose(result.put_weights, [14.0, 9.0])
        np.testing.assert_allclose(result.call_weights, [6.0, 8.0])

    def test_eligible_strike_selection_matches_market_filters(self) -> None:
        puts = select_eligible_option_strikes(
            np.array([2.0, 5.0, 9.0, 10.0]),
            np.array([20.0, 5.0, 20.0, 30.0]),
            minimum_volume=10.0,
            minimum_price=2.0,
            maximum_price=9.0,
            initial_price=10.0,
            is_put=True,
        )
        calls = select_eligible_option_strikes(
            np.array([10.0, 11.0, 15.0, 20.0]),
            np.array([50.0, 20.0, 5.0, 30.0]),
            minimum_volume=10.0,
            minimum_price=10.0,
            maximum_price=20.0,
            initial_price=10.0,
            is_put=False,
        )

        np.testing.assert_array_equal(puts, [2.0, 9.0])
        np.testing.assert_array_equal(calls, [11.0, 20.0])


class OptionLoaderTests(unittest.TestCase):
    """Check that the loader returns data for a selected expiration."""

    def test_loader_can_fetch_one_expiration(self) -> None:
        expected_expiration = "2026-12-18"
        fake_ticker = SimpleNamespace(
            options=(expected_expiration,),
            option_chain=Mock(return_value=SimpleNamespace(
                calls=pd.DataFrame({"strike": [100.0]}),
                puts=pd.DataFrame({"strike": [100.0]}),
            )),
        )
        with patch("model.loader.yf.Ticker", return_value=fake_ticker):
            expirations = get_option_expirations(" ^spx ")
            calls, puts = create_option_dataframe(" ^spx ", expected_expiration)

        self.assertEqual(expirations, (expected_expiration,))
        self.assertEqual(calls["expiration"].tolist(), [expected_expiration])
        self.assertEqual(puts["expiration"].tolist(), [expected_expiration])

    def test_ticker_market_data_returns_quote_and_expirations(self) -> None:
        expected_expirations = ("2026-12-18", "2027-01-15")
        fake_ticker = SimpleNamespace(
            options=expected_expirations,
            fast_info={"lastPrice": 350.0},
        )
        with patch("model.loader.yf.Ticker", return_value=fake_ticker):
            expirations, last_price = get_ticker_market_data(" ^spx ")

        self.assertEqual(expirations, expected_expirations)
        self.assertEqual(last_price, 350.0)

    def test_ticker_market_data_falls_back_to_latest_close(self) -> None:
        fake_ticker = SimpleNamespace(
            options=("2026-12-18",),
            fast_info={},
            history=Mock(return_value=pd.DataFrame({"Close": [342.0, 348.0]})),
        )
        with patch("model.loader.yf.Ticker", return_value=fake_ticker):
            expirations, last_price = get_ticker_market_data("^SPX")

        self.assertEqual(expirations, ("2026-12-18",))
        self.assertEqual(last_price, 348.0)

    def test_ticker_price_is_available_even_without_option_expirations(self) -> None:
        fake_ticker = SimpleNamespace(
            options=(),
            fast_info={"lastPrice": 350.0},
        )
        with patch("model.loader.yf.Ticker", return_value=fake_ticker):
            expirations, last_price = get_ticker_market_data("^SPX")

        self.assertEqual(expirations, ())
        self.assertEqual(last_price, 350.0)

    def test_loader_rejects_tickers_outside_the_supported_index_list(self) -> None:
        self.assertEqual(
            EUROPEAN_INDEX_TICKERS,
            ["^SPX", "^XSP", "^NDX", "^NQX", "^RUT", "^DJX"],
        )
        unsupported_calls = (
            lambda: get_ticker_market_data("AAPL"),
            lambda: get_option_expirations("AAPL"),
            lambda: create_option_dataframe("AAPL"),
        )

        for loader_call in unsupported_calls:
            with self.subTest(loader_call=loader_call):
                with patch("model.loader.yf.Ticker") as ticker_factory:
                    with self.assertRaisesRegex(ValueError, "Choose one of"):
                        loader_call()
                ticker_factory.assert_not_called()


class ChartPriceRangeTests(unittest.TestCase):
    """Check that a loaded ticker price is centered in the drawing area."""

    def test_price_range_centers_loaded_price_and_discards_old_drawing(self) -> None:
        chart = SimpleNamespace(
            min_price=0.0,
            max_price=200.0,
            raw_points=[(100, 200)],
            canvas=Mock(),
        )

        SmoothPayoffDrawer._center_price_range(chart, 350.0)

        self.assertEqual(chart.min_price, 175.0)
        self.assertEqual(chart.max_price, 525.0)
        self.assertEqual(chart.raw_points, [])
        chart.canvas.delete.assert_any_call("curve")
        chart.canvas.delete.assert_any_call("strike_bounds")

    def test_strike_range_draws_and_labels_both_used_boundaries(self) -> None:
        chart = SimpleNamespace(
            canvas=Mock(),
            x_left=10,
            x_right=110,
            y_top=20,
            y_bottom=220,
            min_price=0.0,
            max_price=100.0,
        )

        SmoothPayoffDrawer._draw_strike_range(chart, 25.0, 75.0)

        chart.canvas.delete.assert_called_once_with("strike_bounds")
        line_calls = [
            call
            for call in chart.canvas.create_line.call_args_list
            if call.kwargs.get("tags") == "strike_bounds"
        ]
        self.assertEqual(len(line_calls), 2)
        self.assertEqual(line_calls[0].args[:4], (35.0, 20, 35.0, 220))
        self.assertEqual(line_calls[1].args[:4], (85.0, 20, 85.0, 220))
        text_labels = [
            call.kwargs["text"]
            for call in chart.canvas.create_text.call_args_list
        ]
        self.assertEqual(
            text_labels,
            ["Eligible min strike: 25", "Eligible max strike: 75"],
        )

    def test_strike_preview_uses_selected_expiration_and_volume_threshold(self) -> None:
        chart = SimpleNamespace(
            _market_option_chain=(
                "^SPX",
                "2026-12-18",
                np.array([90.0, 95.0, 100.0]),
                np.array([20.0, 5.0, 20.0]),
                np.array([100.0, 105.0, 110.0]),
                np.array([20.0, 20.0, 5.0]),
            ),
            minimum_volume_entry=SimpleNamespace(get=lambda: "10"),
            spot_entry=SimpleNamespace(get=lambda: "100"),
            min_price=80.0,
            max_price=120.0,
            raw_points=[(70, 100), (90, 80)],
            canvas=Mock(),
            status=Mock(),
            _draw_strike_range=Mock(),
        )

        SmoothPayoffDrawer._update_strike_range_preview(chart)

        chart._draw_strike_range.assert_called_once_with(90.0, 105.0)
        chart.status.configure.assert_called_once_with(
            text=(
                "Eligible strike range for ^SPX, expiration 2026-12-18 "
                "across the chart's full price range: 90 to 105 "
                "(minimum volume 10 contracts)."
            )
        )


if __name__ == "__main__":
    unittest.main()
