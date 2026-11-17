"""Milestone 2: candlestick pattern detectors, features, and a simple backtester.

Works on daily OR hourly candles: change TIMEFRAME below.

Run:  python patterns_and_backtest.py
Needs: trading.db with candles (fetch_and_plot.py for daily, fetch_hourly.py
       for hourly), plus  pip install numpy pandas plotly

Everything is measured in "bars" (one candle). On daily data a bar is a day;
on hourly data a bar is an hour.
"""
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TIMEFRAME = "1h"   # "1d" = daily candles, "1h" = hourly candles

SETTINGS = {
    "1d": {"table": "candles", "features_table": "features",
           "bars_per_day": 1, "hold_bars": 5, "date_fmt": "%Y-%m-%d"},
    "1h": {"table": "candles_1h", "features_table": "features_1h",
           "bars_per_day": 24, "hold_bars": 24, "date_fmt": "%Y-%m-%d %H:%M"},
}
_cfg = SETTINGS[TIMEFRAME]
TABLE = _cfg["table"]
FEATURES_TABLE = _cfg["features_table"]
BARS_PER_DAY = _cfg["bars_per_day"]
HOLD_BARS = _cfg["hold_bars"]      # how long to hold after a signal, in bars
DATE_FMT = _cfg["date_fmt"]

SYMBOL = "BTC-USD"
DB_PATH = "trading.db"
FEE = 0.001        # 0.1% per trade side (typical exchange fee)


# ---------------------------------------------------------------- data
def load_candles(symbol: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT symbol, date, open, high, low, close, volume FROM {TABLE} "
            "WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,), parse_dates=["date"],
        )
    if df.empty:
        raise SystemExit(f"No {TIMEFRAME} candles for {symbol} in {DB_PATH}. "
                         "Run the fetch script for this timeframe first.")
    return df.reset_index(drop=True)


# ------------------------------------------------- features + patterns
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features. Every one uses only the current and past bars."""
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()

    df["bar_return"] = df["close"].pct_change()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_ratio"] = body / rng                      # 0 = no body, 1 = all body
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    df["volatility_20"] = df["bar_return"].rolling(20).std()
    df["volume_change"] = df["volume"].pct_change()
    return df


def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based candlestick patterns (True/False per bar)."""
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    green = df["close"] > df["open"]
    red = df["close"] < df["open"]
    prev_green, prev_red = green.shift(1, fill_value=False), red.shift(1, fill_value=False)
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)

    # Doji: the body is tiny compared with the full high-low range.
    df["doji"] = df["body_ratio"] <= 0.10

    # Hammer: long lower shadow (2x the body), little upper shadow.
    df["hammer"] = (
        (df["lower_shadow"] * (df["high"] - df["low"]) >= 2 * body)
        & (df["upper_shadow"] <= 0.10)
        & (df["body_ratio"] > 0.05)
    )

    # Bullish engulfing: green candle whose body swallows the previous red body.
    df["bullish_engulfing"] = (
        green & prev_red & (df["open"] <= prev_close) & (df["close"] >= prev_open)
    )

    # Bearish engulfing: red candle whose body swallows the previous green body.
    df["bearish_engulfing"] = (
        red & prev_green & (df["open"] >= prev_close) & (df["close"] <= prev_open)
    )
    return df


def save_features(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = out["date"].dt.strftime(DATE_FMT)
    with sqlite3.connect(DB_PATH) as conn:
        out.to_sql(FEATURES_TABLE, conn, if_exists="replace", index=False)


# ------------------------------------------------------------ backtest
def backtest(df: pd.DataFrame, signal: pd.Series, hold_bars: int = HOLD_BARS,
             fee: float = FEE):
    """Long-only backtest.

    A signal on bar t is seen at the CLOSE of bar t, so the earliest we can
    act is the OPEN of bar t+1. (Buying at bar t's close using a signal that
    needs bar t's close would be a subtle form of lookahead.)
    We hold for `hold_bars` bars, sell at the close, and ignore new signals
    while a trade is open.
    """
    n = len(df)
    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()
    sig = signal.to_numpy().astype(bool).tolist()
    equity = np.ones(n)
    eq, filled, trades, bars_in_market = 1.0, 0, [], 0

    i = 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        entry, exit_ = i + 1, min(i + hold_bars, n - 1)
        entry_price = opens[entry]
        equity[filled:entry] = eq                       # cash bars
        for k in range(entry, exit_ + 1):               # mark to market
            equity[k] = eq * closes[k] / (entry_price * (1 + fee))
        equity[exit_] *= (1 - fee)                      # selling fee
        trade_return = equity[exit_] / eq - 1
        trades.append(trade_return)
        eq = equity[exit_]
        bars_in_market += exit_ - entry + 1
        filled = exit_ + 1
        i = exit_ + 1
    equity[filled:] = eq
    return equity, np.array(trades), bars_in_market / n


def max_drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(((curve - peak) / peak).min())


def summarize(name, equity, trades, exposure) -> dict:
    return {
        "strategy": name,
        "trades": len(trades),
        "win_rate": f"{(trades > 0).mean():.0%}" if len(trades) else "-",
        "avg_trade": f"{trades.mean():+.2%}" if len(trades) else "-",
        "total_return": f"{equity[-1] - 1:+.0%}",
        "max_drawdown": f"{max_drawdown(equity):.0%}",
        "time_in_market": f"{exposure:.0%}",
    }


def random_entry_signal(n: int, n_trades: int, rng, hold_bars: int = HOLD_BARS) -> pd.Series:
    """Random buy signals that lead to about `n_trades` trades over `n` bars.

    Matching the NUMBER OF TRADES (not how often a signal fires) is what makes
    the comparison fair: real signals come in clusters, so they trigger far
    fewer trades than the same number of randomly scattered signals, and
    every extra trade pays fees. This gives the random strategy the same fee
    drag and about the same time in the market as the strategy being tested.
    """
    sig = np.zeros(n, dtype=bool)
    if n_trades > 0:
        idle = max(n / n_trades - hold_bars - 1, 0.0)   # average idle bars between trades
        p = 1.0 / (idle + 1.0)
        i = int(rng.geometric(p)) - 1
        while i < n - 1:
            sig[i] = True
            i += hold_bars + int(rng.geometric(p))      # skip the trade, then wait a random time
    return pd.Series(sig)


def random_returns(df: pd.DataFrame, n_trades: int, runs: int = 200, seed: int = 42,
                   hold_bars: int = HOLD_BARS) -> np.ndarray:
    """Total returns of many random-entry strategies making `n_trades` trades."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        sig = random_entry_signal(len(df), n_trades, rng, hold_bars)
        equity, _, _ = backtest(df, sig, hold_bars)
        out.append(equity[-1] - 1)
    return np.array(out)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    df = add_patterns(add_features(load_candles(SYMBOL)))
    save_features(df)
    print(f"{SYMBOL} ({TIMEFRAME}): {len(df):,} candles, "
          f"{df['date'].min():%Y-%m-%d %H:%M} to {df['date'].max():%Y-%m-%d %H:%M}")
    print("\nPattern counts:")
    print(df[["doji", "hammer", "bullish_engulfing", "bearish_engulfing"]].sum().to_string())

    hold = df["close"].to_numpy() / df["close"].iloc[0]       # buy and hold
    rows = [{**summarize("buy_and_hold", hold, np.array([hold[-1] - 1]), 1.0),
             "beats_random": "-"}]
    curves = {"buy_and_hold": hold}

    for name in ["bullish_engulfing", "hammer"]:
        equity, trades, exposure = backtest(df, df[name])
        rand = random_returns(df, len(trades))
        row = summarize(f"{name} ({HOLD_BARS}-bar hold)", equity, trades, exposure)
        row["beats_random"] = f"{(rand < equity[-1] - 1).mean():.0%} of runs"
        rows.append(row)
        curves[name] = equity
        rows.append({"strategy": "  random entries, median", "trades": "-", "win_rate": "-",
                     "avg_trade": "-", "total_return": f"{np.median(rand):+.0%}",
                     "max_drawdown": "-", "time_in_market": "-",
                     "beats_random": f"5th-95th pct: {np.percentile(rand, 5):+.0%} to {np.percentile(rand, 95):+.0%}"})

    print(f"\nBacktest (fee {FEE:.1%} per side, long-only):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nRandom entries make the same NUMBER of trades as the strategy (same fee drag).")
    print("'beats_random' near 50% = no edge; you want 95%+ and enough trades to trust it.")

    step = BARS_PER_DAY          # plot one point per day so hourly charts stay fast
    fig = go.Figure()
    for label, curve in curves.items():
        fig.add_trace(go.Scatter(x=df["date"].iloc[::step], y=curve[::step], name=label))
    fig.update_layout(title="Equity curves (start = 1.0)", yaxis_type="log")
    fig.write_html("equity.html")
    fig.show()