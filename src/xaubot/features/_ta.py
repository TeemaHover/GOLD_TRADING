"""Causal technical-indicator primitives.

Every function here answers "what was the value of X at bar i, using bars
<= i?". None of them centre a window, look forward, or use a statistic computed
over the whole series.

Deliberately not using a third-party TA library: several popular ones centre
windows or backfill leading NaNs by default, which is exactly the silent
look-ahead this system exists to avoid. These are small enough to read and
verify, and each has a known-answer test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaubot.features.base import EPS


# ---------------------------------------------------------------------------
# Moving averages and smoothing
# ---------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (standard 2/(n+1) smoothing)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period, min_periods=period).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/n), used by ATR, RSI, ADX."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_zscore(series: pd.Series, period: int) -> pd.Series:
    """Z-score against the trailing window only."""
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return (series - mean) / (std + EPS)


def rolling_percentile(series: pd.Series, period: int) -> pd.Series:
    """Rank of the current value within its trailing window, in ``[0, 1]``.

    Trailing-window ranking, never a global min/max: normalising against
    statistics of the full series leaks the future into every early bar.
    """
    return series.rolling(period, min_periods=max(20, period // 10)).rank(pct=True)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range."""
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range (Wilder)."""
    return wilder(true_range(high, low, close), period)


def realized_vol(close: pd.Series, period: int, bars_per_day: int) -> pd.Series:
    """Annualised-to-daily realised volatility of log returns."""
    returns = np.log(close).diff()
    return returns.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(bars_per_day)


def parkinson_vol(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    """Parkinson high-low volatility estimator.

    More efficient than close-to-close when the bar range is informative, which
    it is for gold at 5m.
    """
    log_hl = np.log((high / (low + EPS)).clip(lower=1e-9)) ** 2
    return np.sqrt(log_hl.rolling(period, min_periods=period).mean() / (4.0 * np.log(2.0)))


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Garman-Klass volatility estimator (uses the full OHLC of each bar)."""
    log_hl = np.log((high / (low + EPS)).clip(lower=1e-9)) ** 2
    log_co = np.log((close / (open_ + EPS)).clip(lower=1e-9)) ** 2
    estimator = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(estimator.rolling(period, min_periods=period).mean().clip(lower=0.0))


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index (Wilder smoothing)."""
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), period)
    loss = wilder((-delta).clip(lower=0.0), period)
    rs = gain / (loss + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line minus its signal line)."""
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator ``(%K, %D)``."""
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    k = 100.0 * (close - lowest) / (highest - lowest + EPS)
    return k, k.rolling(smooth, min_periods=smooth).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity channel index."""
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period, min_periods=period).mean()
    deviation = (typical - mean).abs().rolling(period, min_periods=period).mean()
    return (typical - mean) / (0.015 * deviation + EPS)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R, in ``[-100, 0]``."""
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100.0 * (highest - close) / (highest - lowest + EPS)


# ---------------------------------------------------------------------------
# Trend strength
# ---------------------------------------------------------------------------
def directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX system, returning ``(adx, di_plus, di_minus)``."""
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    atr_w = wilder(true_range(high, low, close), period)
    di_plus = 100.0 * wilder(plus_dm, period) / (atr_w + EPS)
    di_minus = 100.0 * wilder(minus_dm, period) / (atr_w + EPS)

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus + EPS)
    return wilder(dx, period), di_plus, di_minus


def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: net move divided by total path length.

    Near 1 in a clean trend, near 0 in chop. One of the few trend measures that
    separates "moved a lot" from "went somewhere".
    """
    net = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(period, min_periods=period).sum()
    return net / (path + EPS)


def rolling_linreg(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Rolling least-squares fit against bar index, returning ``(slope, r2)``.

    Closed form rather than an iterative fit: the x values are a fixed
    ``0..n-1`` ramp, so the denominators are constants.
    """
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    y_mean = series.rolling(period, min_periods=period).mean()
    xy = series.rolling(period, min_periods=period).apply(lambda a: float(np.dot(a, x)), raw=True)

    covariance = xy - period * x_mean * y_mean
    slope = covariance / x_var

    y_var = series.rolling(period, min_periods=period).var(ddof=0) * period
    r2 = (covariance**2) / (x_var * y_var + EPS)
    return slope, r2.clip(0.0, 1.0)


def hurst_variance_ratio(close: pd.Series, period: int, lag: int = 8) -> pd.Series:
    """Hurst-like persistence estimate from a variance ratio.

    Named for what it is rather than what it approximates: this is *not* a
    rescaled-range Hurst exponent (too slow to compute per bar). It uses the
    variance-ratio identity ``var(k-step) ~ k^(2H) var(1-step)``, which shares
    the interpretation -- above 0.5 trending, below 0.5 mean-reverting -- while
    being a rolling operation rather than a per-window regression.
    """
    returns = np.log(close).diff()
    var_1 = returns.rolling(period, min_periods=period).var(ddof=0)
    var_k = (np.log(close) - np.log(close).shift(lag)).rolling(period, min_periods=period).var(ddof=0)
    ratio = var_k / (lag * var_1 + EPS)
    return 0.5 + np.log(ratio.clip(lower=1e-6)) / (2.0 * np.log(lag))


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------
def bollinger_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width as a fraction of the middle band."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return (2.0 * num_std * std) / (mid + EPS)


def keltner_width(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 1.5
) -> pd.Series:
    """Keltner channel width as a fraction of the middle band."""
    mid = ema(close, period)
    return (2.0 * mult * atr(high, low, close, period)) / (mid + EPS)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def buy_pressure(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Where the bar closed within its range, mapped to ``[-1, 1]``.

    A crude order-flow proxy. Genuine buy/sell imbalance needs tick or L2 data;
    this is what OHLC alone can honestly support, and it is named as a proxy
    everywhere it appears.
    """
    return ((close - low) - (high - close)) / (high - low + EPS)


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------
def squash(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """Map an unbounded non-negative quantity into ``[0, 1)`` via tanh.

    Used for the continuous pattern-strength features. Preferred over min-max
    scaling because min-max over the full series is leakage, and min-max over a
    trailing window makes the same pattern score differently depending on what
    happened to precede it.
    """
    return np.tanh(series.clip(lower=0.0) / (scale + EPS))


def signed_squash(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """Map an unbounded signed quantity into ``(-1, 1)`` via tanh."""
    return np.tanh(series / (scale + EPS))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields 0 rather than inf on a zero divisor."""
    result = numerator / (denominator.abs() + EPS) * np.sign(denominator).replace(0.0, 1.0)
    return result.replace([np.inf, -np.inf], 0.0)
