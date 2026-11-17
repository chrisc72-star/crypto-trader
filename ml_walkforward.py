"""Milestone 3b: walk-forward machine learning on candle features.

Works on daily OR hourly candles: set TIMEFRAME in patterns_and_backtest.py.

Run:  python ml_walkforward.py
Needs: trading.db and patterns_and_backtest.py in the same folder, plus
       pip install scikit-learn

Question the script answers: can a model that only sees PAST data predict
whether a trade will make money after fees, better than random entries?

How it avoids fooling itself
  * Walk-forward: train on the past, predict the next 90 days, then slide
    forward and retrain. The model never predicts data it was trained on.
  * Purging: the last HOLD_BARS of training rows are dropped, because their
    labels are built from prices inside the test period.
  * Same backtester, same fees, trades start at the NEXT bar's open.
  * Random-entry baseline shown as a distribution, not a single average.
"""
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from patterns_and_backtest import (
    BARS_PER_DAY, DATE_FMT, DB_PATH, FEE, HOLD_BARS, SYMBOL, TIMEFRAME,
    add_features, add_patterns, backtest, load_candles, random_returns, summarize,
)

MIN_TRAIN_BARS = 1000 * BARS_PER_DAY    # first training window (about 2.7 years)
TEST_BLOCK_BARS = 90 * BARS_PER_DAY     # predict this much, then retrain
MARGIN = 0.03           # trade only if P(up) beats the training base rate by this much
RANDOM_RUNS = 300       # random-entry simulations for the baseline

PATTERNS = ["doji", "hammer", "bullish_engulfing", "bearish_engulfing"]
# Look-back windows in bars. On hourly data this adds 1-day and 1-week returns.
HORIZONS = [3, 5, 10, 20] + ([BARS_PER_DAY, 7 * BARS_PER_DAY] if BARS_PER_DAY > 1 else [])
# Time-of-day / day-of-week (hourly only), encoded as circles so 23:00 is next to 00:00.
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"] if BARS_PER_DAY > 1 else []
FEATURES = (
    ["bar_return", "range_pct", "body_ratio", "upper_shadow", "lower_shadow",
     "volatility_20", "volume_change"]
    + [f"ret_{n}" for n in HORIZONS]
    + ["dist_ma20", "dist_ma50", "rsi_14", "volume_ratio_20"]
    + TIME_FEATURES + PATTERNS
)

MODELS = {
    "logistic": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
    "grad_boost": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50,
        l2_regularization=1.0, random_state=0),
}


# ------------------------------------------------------- features + label
def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra features. Each uses only data up to and including the current bar's close."""
    df = df.copy()
    close = df["close"]
    for n in HORIZONS:
        df[f"ret_{n}"] = close.pct_change(n)
    df["dist_ma20"] = close / close.rolling(20).mean() - 1
    df["dist_ma50"] = close / close.rolling(50).mean() - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100.0   # 14 bars with no down moves
    df.loc[(loss == 0) & (gain == 0), "rsi_14"] = 50.0   # completely flat
    df["volume_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()
    if TIME_FEATURES:
        hour, dow = df["date"].dt.hour, df["date"].dt.dayofweek
        df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    for p in PATTERNS:
        df[p] = df[p].astype(int)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """target = 1 if a trade entered at the next bar's open and sold at the close
    HOLD_BARS later makes money AFTER fees. This matches the backtester exactly.
    The label looks into the future on purpose; it is never used as a feature."""
    df = df.copy()
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-HOLD_BARS)
    net = exit_ / entry * (1 - FEE) / (1 + FEE) - 1
    df["target"] = np.where(net.isna(), np.nan, (net > 0).astype(float))
    return df


# ------------------------------------------------------------ walk-forward
def walk_forward(df: pd.DataFrame, make_model):
    """Return out-of-sample P(up), per-row trade thresholds, first test row."""
    n = len(df)
    X = df[FEATURES].to_numpy()
    y = df["target"].to_numpy()
    valid = ~np.isnan(X).any(axis=1)
    first_valid = int(np.argmax(valid))

    proba = np.full(n, np.nan)
    thresh = np.full(n, np.nan)
    first_test = first_valid + MIN_TRAIN_BARS
    s = first_test
    while s < n:
        e = min(s + TEST_BLOCK_BARS, n)
        train = np.arange(first_valid, s - HOLD_BARS + 1)   # purge the last HOLD_BARS
        train = train[valid[train]]                          # skip rows with missing features
        model = make_model()
        model.fit(X[train], y[train])
        test = np.arange(s, e)
        test = test[valid[test]]                             # no prediction if features missing
        if len(test):
            proba[test] = model.predict_proba(X[test])[:, 1]
        thresh[s:e] = y[train].mean() + MARGIN
        s = e
    return proba, thresh, first_test


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    df = add_ml_features(add_label(add_patterns(add_features(load_candles(SYMBOL)))))
    if len(df) < MIN_TRAIN_BARS + TEST_BLOCK_BARS + 100:
        raise SystemExit(f"Only {len(df):,} candles; need at least "
                         f"{MIN_TRAIN_BARS + TEST_BLOCK_BARS + 100:,}. Fetch more history.")

    results, all_proba, curves, pred_rows = [], {}, {}, []
    first_test = None
    for name, make_model in MODELS.items():
        print(f"Training {name} (walk-forward)...", flush=True)
        proba, thresh, first_test = walk_forward(df, make_model)
        all_proba[name] = (proba, thresh)

    ev = df.iloc[first_test:].reset_index(drop=True)          # evaluation window
    print(f"\n{SYMBOL} ({TIMEFRAME}): {len(df):,} candles. Out-of-sample window: "
          f"{ev['date'].min():%Y-%m-%d} to {ev['date'].max():%Y-%m-%d} ({len(ev):,} candles)")
    print(f"Retrained every {TEST_BLOCK_BARS:,} bars on all earlier data; "
          f"fee {FEE:.1%} per side; hold {HOLD_BARS} bars.\n")

    hold = ev["close"].to_numpy() / ev["close"].iloc[0]
    row = summarize("buy_and_hold", hold, np.array([hold[-1] - 1]), 1.0)
    row["beats_random"] = "-"
    results.append(row)
    curves["buy_and_hold"] = hold

    print("Prediction quality (out-of-sample only):")
    for name, (proba, thresh) in all_proba.items():
        p = pd.Series(proba[first_test:])
        y = ev["target"]
        ok = p.notna() & y.notna()
        auc = roc_auc_score(y[ok], p[ok])
        sig = (p > pd.Series(thresh[first_test:])).fillna(False)
        hit = y[ok & sig].mean() if (ok & sig).any() else float("nan")
        print(f"  {name:11s} AUC {auc:.3f} (0.500 = coin flip) | base rate P(up) {y[ok].mean():.1%}"
              f" | P(up) when it signals {hit:.1%} on {int((ok & sig).sum()):,} signals")

        equity, trades, exposure = backtest(ev, sig)
        rand = random_returns(ev, len(trades), RANDOM_RUNS)   # same number of trades = same fee drag
        row = summarize(f"{name} ({HOLD_BARS}-bar hold)", equity, trades, exposure)
        row["beats_random"] = f"{(rand < equity[-1] - 1).mean():.0%} of runs"
        results.append(row)
        row = {"strategy": "  random entries, median", "trades": "-", "win_rate": "-",
               "avg_trade": "-", "total_return": f"{np.median(rand):+.0%}",
               "max_drawdown": "-", "time_in_market": "-",
               "beats_random": f"5th-95th pct: {np.percentile(rand, 5):+.0%} to {np.percentile(rand, 95):+.0%}"}
        results.append(row)
        curves[name] = equity
        for d, pr in zip(ev["date"], p):
            if not np.isnan(pr):
                pred_rows.append((d.strftime(DATE_FMT), name, float(pr)))

    print("\nBacktest on the out-of-sample window:")
    print(pd.DataFrame(results).to_string(index=False))
    print("\n'beats_random' = share of random-entry runs (same NUMBER of trades) the model beat.")
    print("Around 50% means no edge; you want it near 95%+ AND an AUC clearly above 0.5.")

    table = "ml_predictions" if TIMEFRAME == "1d" else f"ml_predictions_{TIMEFRAME}"
    with sqlite3.connect(DB_PATH) as conn:
        pd.DataFrame(pred_rows, columns=["date", "model", "p_up"]).to_sql(
            table, conn, if_exists="replace", index=False)

    step = BARS_PER_DAY          # plot one point per day so hourly charts stay fast
    fig = go.Figure()
    for label, curve in curves.items():
        fig.add_trace(go.Scatter(x=ev["date"].iloc[::step], y=curve[::step], name=label))
    fig.update_layout(title="Out-of-sample equity curves (start = 1.0)", yaxis_type="log")
    fig.write_html("equity_ml.html")
    fig.show()