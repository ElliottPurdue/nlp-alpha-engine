"""Stage 3: ingest market data and rebuild the feature matrix.

Upserts OHLCV into `prices`, then regenerates `daily_features` from scratch.
The rebuild is deliberate: feature definitions change constantly during
development, and a matrix holding rows from two different definitions is worse
than no matrix, because nothing about it looks wrong.
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import database as db

warnings.filterwarnings("ignore")

# Deeper than the sentiment history needs. Prices are backfillable at any time, so
# there is no cost to holding more than the model currently uses.
PRICE_PERIOD = "10y"

# Long horizon in sessions, for H2.
LONG_HORIZON = 5

# Prior observations used to establish what is normal for a ticker. These are
# per-ticker news days, not calendar days, so for thinly covered names the window
# spans well over twenty sessions.
BASELINE_WINDOW = 20

# Below this many prior observations, surprise columns stay NULL rather than being
# computed off one or two points.
MIN_BASELINE = 5


def _to_int(value):
    return None if pd.isna(value) else int(value)


def _to_float(value):
    return None if pd.isna(value) else float(value)


def ingest_prices(conn, tickers, period=PRICE_PERIOD):
    """Fetch and upsert OHLCV per ticker, isolating failures like the scraper does."""
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
    """Sentiment summed per (ticker, session_date).

    Only the sum and count are aggregated here. The mean is taken after the
    calendar roll below, which can merge two session dates onto one trading day;
    averaging two means would misweight them.
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
    """Prices with one-day and five-day forward returns attached."""
    prices = pd.read_sql_query(
        "SELECT ticker, date, close, volume FROM prices ORDER BY ticker, date",
        conn,
        parse_dates=["date"],
    )
    if prices.empty:
        return prices

    # Close-to-close: entered at session D's close, exited at D+1's. This is the
    # horizon session_date was built to respect.
    prices["fwd_return"] = (
        prices.groupby("ticker")["close"].transform(lambda c: c.shift(-1) / c - 1)
    )
    prices["target"] = np.where(
        prices["fwd_return"].isna(), np.nan,
        np.where(prices["fwd_return"] > 0, 1, 0),
    )

    # Consecutive values of this column overlap by four of their five days, so its
    # significance tests need a correction the one-day column does not.
    prices["fwd_return_5d"] = (
        prices.groupby("ticker")["close"]
        .transform(lambda c: c.shift(-LONG_HORIZON) / c - 1)
    )
    return prices


def rebuild_features(conn, model_name=db.FINBERT_MODEL):
    """Regenerate daily_features and return the row count written."""
    sentiment = _load_daily_sentiment(conn, model_name)
    prices = _load_prices(conn)

    if sentiment.empty or prices.empty:
        print("  Both sentiment and prices are required; nothing written.")
        db.replace_daily_features(conn, [])
        return 0

    # session_date is derived without an exchange calendar and can land on a
    # holiday. A forward merge_asof advances each one to the next date that
    # actually traded, which covers holidays without maintaining a calendar.
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

    # A holiday and the day after it can now point at the same trading day, so
    # totals are recombined before the mean is taken.
    daily = (
        rolled.groupby(["ticker", "date"], as_index=False)
        .agg(sum_sentiment=("sum_sentiment", "sum"),
             headline_count=("headline_count", "sum"))
    )
    daily["mean_sentiment"] = daily["sum_sentiment"] / daily["headline_count"]

    merged = daily.merge(prices, on=["ticker", "date"], how="inner")

    # Volume spans 799x across this universe and headline counts 43x, so raw values
    # identify the company rather than describe the day. Ranking within the session
    # makes them comparable.
    for source, rank_name in (("mean_sentiment", "sentiment_rank"),
                              ("headline_count", "headline_rank"),
                              ("volume", "volume_rank")):
        merged[rank_name] = merged.groupby("date")[source].rank(pct=True)

    # Surprise features. Ranks say whether a stock's news is good relative to its
    # peers; these say whether it is unusual relative to the stock's own history.
    # Four articles is a large event for a quiet name and nothing for a busy one.
    #
    # shift(1) before the window is what keeps this causal. Without it each row is
    # measured against a baseline containing itself.
    merged = merged.sort_values(["ticker", "date"])
    by_ticker = merged.groupby("ticker")

    sentiment_baseline = by_ticker["mean_sentiment"].transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=MIN_BASELINE).mean()
    )
    attention_baseline = by_ticker["headline_count"].transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=MIN_BASELINE).median()
    )

    merged["sentiment_surprise"] = merged["mean_sentiment"] - sentiment_baseline
    # Log ratio, so 2 to 8 articles registers the same as 20 to 80. The +1 keeps it
    # finite when a baseline is zero.
    merged["attention_surprise"] = np.log(
        (merged["headline_count"] + 1) / (attention_baseline + 1)
    )

    for source, rank_name in (("sentiment_surprise", "sentiment_surprise_rank"),
                              ("attention_surprise", "attention_surprise_rank")):
        merged[rank_name] = merged.groupby("date")[source].rank(pct=True)

    # Over half the variance of a stock's daily return is the market moving, which
    # company-level sentiment cannot forecast. Subtracting the session mean leaves
    # the part a stock-selection signal could plausibly explain.
    for source, excess_name, target_name in (
        ("fwd_return", "excess_return", "target_relative"),
        ("fwd_return_5d", "excess_return_5d", "target_relative_5d"),
    ):
        merged[excess_name] = (
            merged[source] - merged.groupby("date")[source].transform("mean")
        )
        merged[target_name] = np.where(
            merged[excess_name].isna(), np.nan,
            np.where(merged[excess_name] > 0, 1, 0),
        )

    rows = [
        (row.ticker, row.date.date().isoformat(), float(row.mean_sentiment),
         float(row.sum_sentiment), int(row.headline_count), float(row.close),
         _to_int(row.volume), _to_float(row.sentiment_rank),
         _to_float(row.headline_rank), _to_float(row.volume_rank),
         _to_float(row.sentiment_surprise), _to_float(row.attention_surprise),
         _to_float(row.sentiment_surprise_rank),
         _to_float(row.attention_surprise_rank),
         _to_float(row.fwd_return), _to_float(row.excess_return),
         _to_int(row.target), _to_int(row.target_relative),
         _to_float(row.fwd_return_5d), _to_float(row.excess_return_5d),
         _to_int(row.target_relative_5d))
        for row in merged.itertuples()
    ]
    return db.replace_daily_features(conn, rows)


def run():
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
