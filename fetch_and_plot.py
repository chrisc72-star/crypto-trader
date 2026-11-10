"""Milestone 1: download crypto candles, store them in SQLite, plot them.

Run:  python fetch_and_plot.py
Needs: pip install yfinance pandas plotly
"""
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

SYMBOL = "BTC-USD"
START = "2018-01-01"
DB_PATH = "trading.db"


def fetch_candles(symbol: str, start: str) -> pd.DataFrame:
    """Download daily OHLCV candles and return a clean DataFrame."""
    df = yf.download(symbol, start=start, interval="1d",
                     auto_adjust=False, progress=False)
    # Newer yfinance versions return two-level column names; flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]].dropna()


def save_candles(df: pd.DataFrame, symbol: str) -> None:
    """Replace this symbol's rows in the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS candles (
                   symbol TEXT, date TEXT,
                   open REAL, high REAL, low REAL, close REAL, volume REAL,
                   PRIMARY KEY (symbol, date))"""
        )
        conn.execute("DELETE FROM candles WHERE symbol = ?", (symbol,))
        df.to_sql("candles", conn, if_exists="append", index=False)


def load_candles(symbol: str) -> pd.DataFrame:
    """Read candles back out of the database with SQL."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM candles WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,), parse_dates=["date"],
        )


def plot_candles(df: pd.DataFrame, symbol: str) -> None:
    fig = go.Figure(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
    ))
    fig.update_layout(title=f"{symbol} daily candles", xaxis_rangeslider_visible=False)
    fig.write_html("candles.html")
    fig.show()


if __name__ == "__main__":
    candles = fetch_candles(SYMBOL, START)
    save_candles(candles, SYMBOL)
    stored = load_candles(SYMBOL)
    print(f"Stored {len(stored)} candles for {SYMBOL}")
    print(stored.tail())
    plot_candles(stored, SYMBOL)