"""Diagnostics: what is the model really learning, and can fee drag be reduced?

Run:  python diagnostics.py
Needs: trading.db, patterns_and_backtest.py and ml_walkforward.py in the same folder.
Takes a few minutes on hourly data, because it retrains the models many times.

PART 1 - What is the model learning?
  The label "the trade clears the fees" is easier to hit when the market is moving
  a lot, so a model can score above 0.5 AUC just by spotting volatile periods,
  without any idea which way price will go. This part tests that by
    (a) training on different groups of features, and
    (b) also scoring the plain label "did the price go up?" (no fees).
  Rule of thumb: if "volatility/volume only" matches "all features" on the fees
  label, and the "price goes up" AUCs sit near 0.50, it is predicting volatility,
  not direction.

PART 2 - Can fee drag be reduced?
  Re-runs the backtest with stricter confidence thresholds and different holding
  times. Each result is shown for the first and second half of the test period, so
  you can see whether a setting is consistently good or just lucky once.
  WARNING: this is many tries. The best-looking row is an optimistic pick, so
  only trust a setting that is good in BOTH halves and has plenty of trades.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import ml_walkforward as mw
from ml_walkforward import FEATURES, MODELS, add_label, add_ml_features, walk_forward
from patterns_and_backtest import (
    BARS_PER_DAY, FEE, HOLD_BARS, SYMBOL, TIMEFRAME,
    add_features, add_patterns, backtest, load_candles, random_returns, summarize,
)

DIAG_RANDOM_RUNS = 100                      # random-entry runs per backtest row
MARGINS = [0.00, 0.03, 0.06, 0.09, 0.12]    # how far above the base rate P(up) must be to trade
HOLDS = [12, 24, 48, 72] if BARS_PER_DAY > 1 else [3, 5, 10, 20]
HOLD_MARGINS = [0.03, 0.08]
MIN_TRADES = 30                             # fewer trades than this tells you nothing

VOLATILITY = ["range_pct", "volatility_20", "volume_ratio_20", "volume_change"]
FEATURE_SETS = {
    "all features": FEATURES,
    "volatility/volume only": VOLATILITY,
    "price shape/momentum only": [f for f in FEATURES if f not in VOLATILITY + mw.TIME_FEATURES],
}
if mw.TIME_FEATURES:
    FEATURE_SETS["time of day/week only"] = mw.TIME_FEATURES

COLS = ["strategy", "trades", "win_rate", "avg_trade", "avg_1st_half", "avg_2nd_half",
        "total_return", "max_drawdown", "time_in_market", "beats_random"]


def auc_of(proba, y, first_test) -> float:
    p, t = pd.Series(proba[first_test:]), pd.Series(y[first_test:])
    ok = p.notna() & t.notna()
    return roc_auc_score(t[ok], p[ok])


def bucket_table(ev: pd.DataFrame, score: np.ndarray, title: str) -> None:
    """Split out-of-sample rows into 5 equal groups by `score` and compare outcomes."""
    d = pd.DataFrame({"score": score, "clears_fees": ev["target"].to_numpy(),
                      "price_up": ev["target_raw"].to_numpy(), "ret": ev["fwd_ret"].to_numpy()}).dropna()
    d["bucket"] = pd.qcut(d["score"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    g = d.groupby("bucket", observed=True).agg(
        rows=("ret", "size"), clears_fees=("clears_fees", "mean"), price_up=("price_up", "mean"),
        avg_return=("ret", "mean"), avg_abs_move=("ret", lambda s: s.abs().mean()))
    for col in ("clears_fees", "price_up"):
        g[col] = g[col].map("{:.1%}".format)
    for col in ("avg_return", "avg_abs_move"):
        g[col] = g[col].map("{:+.2%}".format)
    print(title)
    print(g.to_string(), "\n")


def evaluate(ev: pd.DataFrame, sig: pd.Series, hold: int, label: str) -> dict:
    equity, trades, exposure = backtest(ev, sig, hold)
    row = summarize(label, equity, trades, exposure)
    if len(trades) < MIN_TRADES:
        row.update(beats_random="too few trades", avg_1st_half="-", avg_2nd_half="-")
        return row
    half = len(ev) // 2
    for key, (a, b) in {"avg_1st_half": (0, half), "avg_2nd_half": (half, len(ev))}.items():
        _, t, _ = backtest(ev.iloc[a:b].reset_index(drop=True),
                           sig.iloc[a:b].reset_index(drop=True), hold)
        row[key] = f"{t.mean():+.2%}" if len(t) else "-"
    rand = random_returns(ev, len(trades), DIAG_RANDOM_RUNS, hold_bars=hold)
    row["beats_random"] = f"{(rand < equity[-1] - 1).mean():.0%} of runs"
    return row


if __name__ == "__main__":
    features_df = add_ml_features(add_patterns(add_features(load_candles(SYMBOL))))
    df = add_label(features_df)                       # default holding time
    saved, results1, first_test = {}, [], None

    # ---------------------------------------------------------------- PART 1
    print(f"{SYMBOL} ({TIMEFRAME}), fee {FEE:.1%} per side. Training models "
          f"(this takes a few minutes)...", flush=True)
    for model_name, make_model in MODELS.items():
        for set_name, feats in FEATURE_SETS.items():
            print(f"  {model_name}: {set_name}", flush=True)
            p_fee, base_fee, first_test = walk_forward(df, make_model, features=feats, target="target")
            p_raw, _, _ = walk_forward(df, make_model, features=feats, target="target_raw")
            results1.append({
                "model": model_name, "features": set_name,
                "AUC: clears fees": f"{auc_of(p_fee, df['target'].to_numpy(), first_test):.3f}",
                "AUC: price goes up": f"{auc_of(p_raw, df['target_raw'].to_numpy(), first_test):.3f}",
            })
            if set_name == "all features":
                saved[model_name] = (p_fee, base_fee)

    ev = df.iloc[first_test:].reset_index(drop=True)
    print(f"\nOut-of-sample window: {ev['date'].min():%Y-%m-%d} to {ev['date'].max():%Y-%m-%d} "
          f"({len(ev):,} candles)")
    print(f"\n{'=' * 70}\nPART 1: WHAT IS THE MODEL LEARNING?\n{'=' * 70}")
    print(f"Base rates in the test window: {ev['target'].mean():.1%} of {HOLD_BARS}-bar trades clear "
          f"the fees; price rises in {ev['target_raw'].mean():.1%}.")
    print("(AUC 0.500 = coin flip)\n")
    print(pd.DataFrame(results1).to_string(index=False))
    print("\nIf 'volatility/volume only' ~ 'all features' on the fees label, while the "
          "'price goes up' AUCs\nare near 0.50, the model is mostly spotting volatile periods, "
          "not predicting direction.\n")

    for model_name, (proba, _) in saved.items():
        bucket_table(ev, proba[first_test:],
                     f"{model_name}: outcomes by model confidence (Q1 = least confident, Q5 = most)")
    bucket_table(ev, ev["volatility_20"].to_numpy(),
                 "Outcomes by trailing volatility (Q1 = calmest, Q5 = wildest), no model involved")
    print("Read the tables: if avg_abs_move rises with confidence/volatility but price_up barely "
          "moves,\nthe 'edge' is about the SIZE of moves, not their direction.\n")

    # ---------------------------------------------------------------- PART 2
    print(f"{'=' * 70}\nPART 2: CAN FEE DRAG BE REDUCED?\n{'=' * 70}")
    print(f"A round trip costs about {2 * FEE:.1%}. 'margin' = how far above the base rate the "
          "model's P(up)\nmust be before it trades. Higher margin = fewer, more selective trades.\n")

    rows = [{**summarize("buy_and_hold", ev["close"].to_numpy() / ev["close"].iloc[0],
                         np.array([ev["close"].iloc[-1] / ev["close"].iloc[0] - 1]), 1.0),
             "avg_1st_half": "-", "avg_2nd_half": "-", "beats_random": "-"}]
    print(f"2a. Confidence threshold sweep ({HOLD_BARS}-bar hold)")
    for model_name, (proba, base) in saved.items():
        p, b = pd.Series(proba[first_test:]), pd.Series(base[first_test:])
        for m in MARGINS:
            sig = (p > b + m).fillna(False)
            rows.append(evaluate(ev, sig, HOLD_BARS, f"{model_name} margin={m:.2f}"))
    print(pd.DataFrame(rows)[COLS].to_string(index=False), "\n")

    print("2b. Holding time sweep (models retrained for each holding time)", flush=True)
    rows = []
    for h in HOLDS:
        dfh = add_label(features_df, hold=h)
        for model_name, make_model in MODELS.items():
            print(f"  hold={h}: {model_name}", flush=True)
            proba, base, _ = walk_forward(dfh, make_model, hold=h)
            p, b = pd.Series(proba[first_test:]), pd.Series(base[first_test:])
            for m in HOLD_MARGINS:
                sig = (p > b + m).fillna(False)
                rows.append(evaluate(ev, sig, h, f"{model_name} hold={h} margin={m:.2f}"))
    print(pd.DataFrame(rows)[COLS].to_string(index=False), "\n")

    print("How to read Part 2:")
    print("  * avg_trade must clearly beat zero AFTER fees, in BOTH halves, with 100+ trades.")
    print("  * 'beats_random' near 50% = no edge; random entries lose to fees, so beating them is easy.")
    print("  * Compare total_return with buy_and_hold at the top of 2a. A strategy that earns less "
          "with a similar\n    drawdown has not added value.")
    print("  * You just looked at dozens of settings. The best one is partly luck by construction.")