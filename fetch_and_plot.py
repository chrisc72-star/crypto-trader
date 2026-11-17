"""Milestone 3a: download HOURLY crypto candles with ccxt and store them in SQLite.

Run:  python fetch_hourly.py
Needs: pip install ccxt

Safe to re-run: it resumes from the last candle already stored, so later runs
only download what's new.

Notes
  * Candles go into table `candles_1h` (the daily table `candles` is untouched).
  * Timestamps are UTC.
  * The newest candle is skipped until the hour has finished, so you never
    store a half-formed candle.
  * Exchange choice: Bitstamp has free public history back to 2011 and is
    reachable from the US. Binance.com blocks US users; if you want
    USDT pairs, try EXCHANGE = "binanceus" and PAIR = "BTC/USDT" (history
    starts in late 2019).
"""
import sqlite3
import time
from datetime import datetime, timezone

import ccxt

EXCHANGE = "bitstamp"
PAIR = "BTC/USD"
DB_SYMBOL = "BTC-USD"        # name used inside your database
START = "2018-01-01"         # only used on the very first run
DB_PATH = "trading.db"
TIMEFRAME = "1h"
LIMIT = 1000                 # candles per request
HOUR_MS = 3_600_000


def fmt(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ensure_table() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS candles_1h (
                   symbol TEXT, ts INTEGER, date TEXT,
                   open REAL, high REAL, low REAL, close REAL, volume REAL,
                   PRIMARY KEY (symbol, ts))"""
        )


def resume_point() -> int:
    """Where to start downloading: just after the newest stored candle."""
    with sqlite3.connect(DB_PATH) as conn:
        newest = conn.execute(
            "SELECT MAX(ts) FROM candles_1h WHERE symbol = ?", (DB_SYMBOL,)
        ).fetchone()[0]
    if newest is not None:
        print(f"Resuming after {fmt(newest)} UTC")
        return newest + HOUR_MS
    start = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"First run: downloading from {START}")
    return int(start.timestamp() * 1000)


def fetch_all(ex, since_ms: int, until_ms: int) -> list:
    """Page through the exchange's history, oldest to newest."""
    rows, since, retries, batches = [], since_ms, 0, 0
    while since < until_ms:
        try:
            batch = ex.fetch_ohlcv(PAIR, TIMEFRAME, since=since, limit=LIMIT)
        except ccxt.NetworkError as err:          # timeouts, dropped connections
            retries += 1
            if retries > 5:
                raise
            print(f"  network hiccup ({err.__class__.__name__}); retry {retries}/5")
            time.sleep(2 * retries)
            continue
        retries = 0
        batch = [c for c in batch if c[0] >= since]   # guarantees we always move forward
        if not batch:
            since += LIMIT * HOUR_MS                  # nothing in this stretch; skip ahead
            continue
        rows.extend(batch)
        since = batch[-1][0] + HOUR_MS
        batches += 1
        if batches % 10 == 0:
            print(f"  ...{fmt(batch[-1][0])} UTC  ({len(rows):,} candles so far)")
    return rows


def save(rows: list) -> int:
    """Store finished candles; returns how many rows the table gained."""
    now_ms = int(time.time() * 1000)
    clean = {}
    for ts, o, h, l, c, v in rows:
        if ts + HOUR_MS > now_ms:          # hour not finished yet
            continue
        if None in (o, h, l, c):
            continue
        clean[int(ts)] = (DB_SYMBOL, int(ts), fmt(ts), o, h, l, c, v or 0.0)
    with sqlite3.connect(DB_PATH) as conn:
        before = conn.execute("SELECT COUNT(*) FROM candles_1h").fetchone()[0]
        conn.executemany(
            "INSERT OR REPLACE INTO candles_1h VALUES (?,?,?,?,?,?,?,?)", list(clean.values()))
        after = conn.execute("SELECT COUNT(*) FROM candles_1h").fetchone()[0]
    return after - before


def report() -> None:
    """Sanity checks: how much data, and are there holes in it?"""
    with sqlite3.connect(DB_PATH) as conn:
        ts = [r[0] for r in conn.execute(
            "SELECT ts FROM candles_1h WHERE symbol = ? ORDER BY ts", (DB_SYMBOL,))]
        bad = conn.execute(
            "SELECT COUNT(*) FROM candles_1h WHERE symbol = ? AND (high < low OR close <= 0 OR open <= 0)",
            (DB_SYMBOL,)).fetchone()[0]
    if not ts:
        print("No candles stored.")
        return
    gaps = [(ts[i], (ts[i + 1] - ts[i]) // HOUR_MS - 1)
            for i in range(len(ts) - 1) if ts[i + 1] - ts[i] != HOUR_MS]
    missing = sum(g for _, g in gaps)
    print(f"\n{DB_SYMBOL} hourly: {len(ts):,} candles, {fmt(ts[0])} to {fmt(ts[-1])} UTC")
    print(f"Missing hours: {missing:,} in {len(gaps)} gaps "
          f"({missing / (len(ts) + missing):.2%} of the period)")
    for start, size in sorted(gaps, key=lambda g: -g[1])[:5]:
        print(f"  largest gaps: {size} hours starting after {fmt(start)}")
    print(f"Rows failing basic price checks: {bad}")


def main() -> None:
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})
    if not ex.has.get("fetchOHLCV"):
        raise SystemExit(f"{EXCHANGE} does not provide candle history through ccxt.")
    ensure_table()
    since = resume_point()
    rows = fetch_all(ex, since, int(time.time() * 1000))
    print(f"Downloaded {len(rows):,} candles from {EXCHANGE}; saved {save(rows):,} new ones.")
    report()


if __name__ == "__main__":
    main()