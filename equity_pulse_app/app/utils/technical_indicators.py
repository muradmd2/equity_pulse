"""Technical indicator calculations for descriptive analysis."""

from math import sqrt
from statistics import mean, stdev
from typing import Any

from app.utils.tracing import traced

TRADING_DAYS_PER_YEAR = 252


@traced("technical_indicators.calculate", run_type="tool")
def calculate_technical_indicators(prices: list[float | int]) -> dict[str, Any]:
    """Calculate descriptive technical indicators from closing prices."""

    closes = [float(price) for price in prices if price is not None]
    latest_close = closes[-1] if closes else None

    sma_20 = simple_moving_average(closes, 20)
    sma_50 = simple_moving_average(closes, 50)
    sma_200 = simple_moving_average(closes, 200)
    macd_data = macd(closes)

    return {
        "latest_close": latest_close,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi_14": rsi(closes, 14),
        "macd": macd_data,
        "volatility": annualized_volatility(closes),
        "support": approximate_support(closes),
        "resistance": approximate_resistance(closes),
        "trend": describe_trend(latest_close, sma_20, sma_50, sma_200, macd_data),
        "data_points": len(closes),
    }


def simple_moving_average(prices: list[float | int], window: int) -> float | None:
    """Return a simple moving average for the latest window."""

    if len(prices) < window:
        return None
    return round(mean(float(price) for price in prices[-window:]), 4)


def rsi(prices: list[float | int], period: int = 14) -> float | None:
    """Return RSI using Wilder smoothing."""

    closes = [float(price) for price in prices]
    if len(closes) <= period:
        return None

    deltas = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [abs(min(delta, 0.0)) for delta in deltas]

    average_gain = mean(gains[:period])
    average_loss = mean(losses[:period])

    for index in range(period, len(deltas)):
        average_gain = ((average_gain * (period - 1)) + gains[index]) / period
        average_loss = ((average_loss * (period - 1)) + losses[index]) / period

    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return round(100 - (100 / (1 + relative_strength)), 4)


def macd(prices: list[float | int], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float | None]:
    """Return MACD line, signal line, and histogram."""

    closes = [float(price) for price in prices]
    if len(closes) < slow:
        return {"macd": None, "signal": None, "histogram": None}

    fast_ema = exponential_moving_average_series(closes, fast)
    slow_ema = exponential_moving_average_series(closes, slow)
    macd_values = [
        fast_value - slow_value
        for fast_value, slow_value in zip(fast_ema, slow_ema, strict=True)
        if fast_value is not None and slow_value is not None
    ]

    if not macd_values:
        return {"macd": None, "signal": None, "histogram": None}

    signal_values = exponential_moving_average_series(macd_values, signal)
    macd_line = macd_values[-1]
    signal_line = signal_values[-1]
    histogram = None if signal_line is None else macd_line - signal_line

    return {
        "macd": round(macd_line, 4),
        "signal": round(signal_line, 4) if signal_line is not None else None,
        "histogram": round(histogram, 4) if histogram is not None else None,
    }


def exponential_moving_average_series(values: list[float], window: int) -> list[float | None]:
    """Return an EMA series seeded with the first full-window SMA."""

    if len(values) < window:
        return [None for _ in values]

    multiplier = 2 / (window + 1)
    series: list[float | None] = [None for _ in range(window - 1)]
    current_ema = mean(values[:window])
    series.append(current_ema)

    for value in values[window:]:
        current_ema = (value - current_ema) * multiplier + current_ema
        series.append(current_ema)

    return series


def annualized_volatility(prices: list[float | int]) -> float | None:
    """Return annualized volatility from daily close-to-close returns."""

    closes = [float(price) for price in prices]
    if len(closes) < 3:
        return None

    returns = [
        (closes[index] / closes[index - 1]) - 1
        for index in range(1, len(closes))
        if closes[index - 1] != 0
    ]
    if len(returns) < 2:
        return None
    return round(stdev(returns) * sqrt(TRADING_DAYS_PER_YEAR), 4)


def approximate_support(prices: list[float | int], window: int = 20) -> float | None:
    """Approximate support as the recent-window low."""

    if not prices:
        return None
    recent_prices = [float(price) for price in prices[-window:]]
    return round(min(recent_prices), 4)


def approximate_resistance(prices: list[float | int], window: int = 20) -> float | None:
    """Approximate resistance as the recent-window high."""

    if not prices:
        return None
    recent_prices = [float(price) for price in prices[-window:]]
    return round(max(recent_prices), 4)


def describe_trend(
    latest_close: float | None,
    sma_20: float | None,
    sma_50: float | None,
    sma_200: float | None,
    macd_data: dict[str, float | None],
) -> str:
    """Describe indicator alignment without making a prediction."""

    if latest_close is None:
        return "unavailable"

    upward_signals = 0
    downward_signals = 0

    for average in (sma_20, sma_50, sma_200):
        if average is None:
            continue
        if latest_close > average:
            upward_signals += 1
        elif latest_close < average:
            downward_signals += 1

    histogram = macd_data.get("histogram")
    if histogram is not None:
        if histogram > 0:
            upward_signals += 1
        elif histogram < 0:
            downward_signals += 1

    if upward_signals > downward_signals:
        return "upward"
    if downward_signals > upward_signals:
        return "downward"
    return "mixed"
