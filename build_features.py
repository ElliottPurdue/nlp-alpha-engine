"""Pipeline stage 3: ingest market data and rebuild the feature matrix.

Two responsibilities:

    1. Upsert daily OHLCV bars from yfinance into `prices`, which is source of
       truth and accumulates over time.
    2. Regenerate `daily_features` in full by joining daily sentiment onto the
       trading calendar implied by `prices`.

The second is a full rebuild rather than an incremental merge. Feature
definitions change repeatedly while a model is under development, and a matrix
containing rows computed under two different definitions is worse than no matrix
at all, because nothing about it looks wrong.
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import database as db

# yfinance surfaces pandas FutureWarnings that are not actionable here.
warnings.filterwarnings("ignore")

# Deeper than current sentiment history can use. Price history is fully
# backfillable at any time, so there is no cost to storing more of it than the
# model presently needs, and it leaves room for a longer backtest later.
PRICE_PERIOD = "2y"


def _to_int(value):
    """Return a Python int, or None where pandas holds a missing value."""
    return None if pd.isna(value) else int(value)


def _to_float(value):
    """Return a Python float, or None where pandas holds a missing value."""
    return None if pd.isna(value) else float(value)


def ingest_prices(conn, tickers, period=PRICE_PERIOD):
    """Fetch and upsert OHLCV bars for each ticker.

    Failures are isolated per symbol, matching the scraper's behaviour: one
    delisted or mistyped ticker must not cost the run every other symbol.

    Returns:
        The number of bars written.
    """
    print(f"Fetching {period} of market data for {len(tickers)} tickers...")
    total = 0
    failed = []

    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as exc:
            print(f"  {ticker}: FAILED ({exc})")
            failed.append(ticker)
            continue

        if history.empty:
            failed.append(ticker)
            continue

        history = history.reset_index()
        date_col = "Date" if "Date" in history.columns else history.columns[0]
        dates = pd.to_datetime(history[date_col]).dt.tz_localize(None).dt.date

        rows = [
            (ticker, day.isoformat(), _to_float(o), _to_float(h),
             _to_float(low), float(close), _to_int(volume))
            for day, o, h, low, close, volume in zip(
                dates, history["Open"], history["High"],
                history["Low"], history["Close"], history["Volume"])
            if not pd.isna(close)
        ]
        total += db.upsert_prices(conn, rows)

    summary = f"  {total} bars written"
    if failed:
        summary += f"; {len(failed)} unavailable: {', '.join(failed)}"
    print(summary)
    return total


def _load_daily_sentiment(conn, model_name):
    """Aggregate sentiment per (ticker, session_date).

    Only the sum and the count are aggregated in SQL. The mean is derived after
    the calendar roll below, because that roll can merge two session dates onto a
    single trading day, and averaging two means would silently misweight them.
    """
    return pd.read_sql_query(
        """
        SELECT t.ticker,
               n.session_date,
               SUM(s.net_sentiment) AS sum_sentiment,
               COUNT(*)             AS headline_count
        FROM raw_news n
        JOIN news_tickers t USING (article_id)
        JOIN sentiment_scores s
          ON s.article_id = n.article_id AND s.model_name = ?
        GROUP BY t.ticker, n.session_date
        """,
        conn,
        params=(model_name,),
        parse_dates=["session_date"],
    )


def _load_prices(conn):
    """Load prices with the forward return and binary target attached."""
    prices = pd.read_sql_query(
        "SELECT ticker, date, close, volume FROM prices ORDER BY ticker, date",
        conn,
        parse_dates=["date"],
    )
    if prices.empty:
        return prices

    # Close-to-close: a position entered at session D's close is exited at the
    # close of D+1, which is the horizon session_date was constructed to respect.
    prices["fwd_return"] = (
        prices.groupby("ticker")["close"].transform(lambda c: c.shift(-1) / c - 1)
    )
    prices["target"] = np.where(
        prices["fwd_return"].isna(), np.nan,
        np.where(prices["fwd_return"] > 0, 1, 0),
    )
    return prices


def rebuild_features(conn, model_name=db.FINBERT_MODEL):
    """Regenerate daily_features from stored sentiment and prices.

    Returns:
        The number of ticker-days written.
    """
    sentiment = _load_daily_sentiment(conn, model_name)
    prices = _load_prices(conn)

    if sentiment.empty or prices.empty:
        print("  Both sentiment and prices are required; nothing written.")
        db.replace_daily_features(conn, [])
        return 0

    # session_date is derived without an exchange calendar, so it can land on a
    # market holiday. merge_asof with direction="forward" advances each one to
    # the next date that actually traded, resolving holidays without hardcoding
    # a calendar that would need maintaining every year.
    calendar = prices[["ticker", "date"]].sort_values("date")
    sentiment = sentiment.sort_values("session_date")
    rolled = pd.merge_asof(
        sentiment,
        calendar,
        left_on="session_date",
        right_on="date",
        by="ticker",
        direction="forward",
    ).dropna(subset=["date"])

    # A holiday and the day following it can now point at the same trading day,
    # so totals are recombined before the mean is taken.
    daily = (
        rolled.groupby(["ticker", "date"], as_index=False)
        .agg(sum_sentiment=("sum_sentiment", "sum"),
             headline_count=("headline_count", "sum"))
    )
    daily["mean_sentiment"] = daily["sum_sentiment"] / daily["headline_count"]

    merged = daily.merge(prices, on=["ticker", "date"], how="inner")

    rows = [
        (row.ticker, row.date.date().isoformat(), float(row.mean_sentiment),
         float(row.sum_sentiment), int(row.headline_count), float(row.close),
         _to_int(row.volume), _to_float(row.fwd_return), _to_int(row.target))
        for row in merged.itertuples()
    ]
    return db.replace_daily_features(conn, rows)


def run():
    """Ingest prices for every tracked ticker, then rebuild the feature matrix."""
    with db.connect() as conn:
        db.init_db(conn)

        tickers = db.distinct_tickers(conn)
        if not tickers:
            print("No tickers in the database yet; run scraper.py first.")
            return

        ingest_prices(conn, tickers)

        print("\nRebuilding daily_features...")
        written = rebuild_features(conn)
        print(f"  {written} ticker-days written")

        trainable = conn.execute(
            "SELECT COUNT(*) FROM daily_features WHERE target IS NOT NULL"
        ).fetchone()[0]
        print(f"  {trainable} carry a forward return and are trainable")

        print("\nDatabase contents:")
        for table, count in db.table_counts(conn).items():
            print(f"  {table:<18} {count:>6} rows")


if __name__ == "__main__":
    run()