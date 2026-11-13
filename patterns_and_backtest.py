"""Milestone 2: candlestick pattern detectors, features, and a simple backtester.

Run:  python patterns_and_backtest.py
Needs: trading.db from fetch_and_plot.py, plus  pip install numpy pandas plotly
"""
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SYMBOL = "BTC-USD"
DB_PATH = "trading.db"
HOLD_DAYS = 5      # how long to hold after a signal
FEE = 0.001        # 0.1% per trade side (typical exchange fee)


# ---------------------------------------------------------------- data
def load_candles(symbol: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM candles WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,), parse_dates=["date"],
        )
    return df.reset_index(drop=True)


# ------------------------------------------------- features + patterns
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features. Every one uses only today's and past data."""
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()

    df["daily_return"] = df["close"].pct_change()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["body_ratio"] = body / rng                      # 0 = no body, 1 = all body
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df["volume_change"] = df["volume"].pct_change()
    return df


def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based candlestick patterns (True/False per day)."""
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
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        out.to_sql("features", conn, if_exists="replace", index=False)


# ------------------------------------------------------------ backtest
def backtest(df: pd.DataFrame, signal: pd.Series, hold_days: int = HOLD_DAYS,
             fee: float = FEE):
    """Long-only backtest.

    A signal on day t is seen at the CLOSE of day t, so the earliest we can
    act is the OPEN of day t+1. (Buying at day t's close using a signal that
    needs day t's close would be a subtle form of lookahead.)
    We hold for `hold_days` days, sell at the close, and ignore new signals
    while a trade is open.
    """
    n = len(df)
    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()
    sig = signal.to_numpy()
    equity = np.ones(n)
    eq, filled, trades, days_in_market = 1.0, 0, [], 0

    i = 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        entry, exit_ = i + 1, min(i + hold_days, n - 1)
        entry_price = opens[entry]
        equity[filled:entry] = eq                       # cash days
        for k in range(entry, exit_ + 1):               # mark to market
            equity[k] = eq * closes[k] / (entry_price * (1 + fee))
        equity[exit_] *= (1 - fee)                      # selling fee
        trade_return = equity[exit_] / eq - 1
        trades.append(trade_return)
        eq = equity[exit_]
        days_in_market += exit_ - entry + 1
        filled = exit_ + 1
        i = exit_ + 1
    equity[filled:] = eq
    return equity, np.array(trades), days_in_market / n


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


def random_baseline(df, frequency, runs=200, seed=42) -> float:
    """Average return of random entries made as often as the real signal."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(runs):
        fake = pd.Series(rng.random(len(df)) < frequency)
        equity, _, _ = backtest(df, fake)
        results.append(equity[-1] - 1)
    return float(np.mean(results))


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    df = add_patterns(add_features(load_candles(SYMBOL)))
    save_features(df)
    print(f"{SYMBOL}: {len(df)} days, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print("\nPattern counts:")
    print(df[["doji", "hammer", "bullish_engulfing", "bearish_engulfing"]].sum().to_string())

    hold = df["close"].to_numpy() / df["close"].iloc[0]       # buy and hold
    rows = [summarize("buy_and_hold", hold, np.array([hold[-1] - 1]), 1.0)]
    curves = {"buy_and_hold": hold}

    for name in ["bullish_engulfing", "hammer"]:
        equity, trades, exposure = backtest(df, df[name])
        rows.append(summarize(f"{name} ({HOLD_DAYS}d hold)", equity, trades, exposure))
        curves[name] = equity
        freq = df[name].mean()
        rand = random_baseline(df, freq)
        rows.append({"strategy": f"  random, same frequency", "trades": "-",
                     "win_rate": "-", "avg_trade": "-",
                     "total_return": f"{rand:+.0%}", "max_drawdown": "-",
                     "time_in_market": "-"})

    print(f"\nBacktest (fee {FEE:.1%} per side, long-only):")
    print(pd.DataFrame(rows).to_string(index=False))

    fig = go.Figure()
    for label, curve in curves.items():
        fig.add_trace(go.Scatter(x=df["date"], y=curve, name=label))
    fig.update_layout(title="Equity curves (start = 1.0)", yaxis_type="log")
    fig.write_html("equity.html")
    fig.show()