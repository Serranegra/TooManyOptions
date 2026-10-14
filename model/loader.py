import math

import pandas as pd
import yfinance as yf

EUROPEAN_INDEX_TICKERS = [
    "^SPX",  # S&P 500 Index Options (standard size and highest liquidity)
    "^XSP",  # Mini-S&P 500 Index Options (one-tenth the size of SPX)
    "^NDX",  # Nasdaq-100 Index Options (technology-focused index)
    "^NQX",  # Mini-Nasdaq-100 Index Options (one-fifth the size of NDX)
    "^RUT",  # Russell 2000 Index Options (small-cap stocks)
    "^DJX",  # Dow Jones Industrial Average Options (one-hundredth DJIA)
]


def _normalize_ticker_symbol(ticker_symbol: str) -> str:
    """Normalize a symbol and reject anything outside the supported index list."""
    normalized_symbol = ticker_symbol.strip().upper()
    if normalized_symbol not in EUROPEAN_INDEX_TICKERS:
        supported_symbols = ", ".join(EUROPEAN_INDEX_TICKERS)
        raise ValueError(
            f"Unsupported ticker '{ticker_symbol}'. Choose one of: "
            f"{supported_symbols}."
        )
    return normalized_symbol


def get_ticker_market_data(
    ticker_symbol: str,
) -> tuple[tuple[str, ...], float]:
    """Return supported index-option expirations and the index's latest price.

    The fast quote is preferred. If it does not contain a usable last price,
    the latest available daily close is used instead.

    Raises:
        ValueError: If the symbol is not in ``EUROPEAN_INDEX_TICKERS`` or no
            usable market price is available.
    """
    ticker_symbol = _normalize_ticker_symbol(ticker_symbol)
    ticker = yf.Ticker(ticker_symbol)
    expirations = tuple(ticker.options)

    try:
        last_price = float(ticker.fast_info.get("lastPrice"))
    except (AttributeError, KeyError, TypeError, ValueError):
        last_price = float("nan")

    if not math.isfinite(last_price) or last_price <= 0:
        history = ticker.history(period="5d")
        if history.empty or "Close" not in history:
            raise ValueError(f"No current market price is available for {ticker_symbol}.")
        closing_prices = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if closing_prices.empty:
            raise ValueError(f"No current market price is available for {ticker_symbol}.")
        last_price = float(closing_prices.iloc[-1])

    if not math.isfinite(last_price) or last_price <= 0:
        raise ValueError(f"No valid market price is available for {ticker_symbol}.")
    return expirations, last_price


def get_option_expirations(ticker_symbol: str) -> tuple[str, ...]:
    """Return expiration dates for one of the supported index tickers.

    Args:
        ticker_symbol: A symbol from ``EUROPEAN_INDEX_TICKERS``.

    Raises:
        ValueError: If the symbol is not in ``EUROPEAN_INDEX_TICKERS``.

    Returns:
        Expiration dates as ISO-formatted strings.
    """
    ticker_symbol = _normalize_ticker_symbol(ticker_symbol)
    ticker = yf.Ticker(ticker_symbol)
    return tuple(ticker.options)


def create_option_dataframe(
    ticker_symbol: str,
    expiration: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load option-chain data for a ticker, optionally for one expiration.

    Args:
        ticker_symbol: A symbol from ``EUROPEAN_INDEX_TICKERS``.
        expiration: Expiration date to load. If omitted, all available dates are
            loaded and concatenated.

    Returns:
        A pair of DataFrames: calls first and puts second. The provider's option
        chain columns are preserved, with an additional ``expiration`` column.
        The ``volume`` column is the number of contracts traded during the day;
        it is not the dollar value or open interest.

    Raises:
        ValueError: If the symbol is not supported or no option expirations
            are available for the ticker.
    """
    ticker_symbol = _normalize_ticker_symbol(ticker_symbol)
    ticker = yf.Ticker(ticker_symbol)
    expirations = (expiration,) if expiration is not None else ticker.options
    if not expirations:
        raise ValueError(f"No option expirations are available for {ticker_symbol}.")

    all_calls = []
    all_puts = []

    for exp in expirations:
        opt = ticker.option_chain(exp)

        calls = opt.calls.copy()
        puts = opt.puts.copy()

        calls["expiration"] = exp
        puts["expiration"] = exp

        all_calls.append(calls)
        all_puts.append(puts)

    df_calls = pd.concat(all_calls, ignore_index=True)
    df_puts = pd.concat(all_puts, ignore_index=True)

    return df_calls, df_puts
