"""Backfill historical headlines from the FNSPID dataset.

Streams FNSPID's headline-level export from HuggingFace in chunks, filters to the
configured universe and date range, and writes through the same upsert path the
live scraper uses. Nothing is stored on disk.

One difference from live collection is material and deliberate. FNSPID records
are largely date-only, carrying no intraday timestamp, so it cannot be
established whether an article preceded or followed the close. Those records are
attributed to the FOLLOWING session, the only assumption that cannot introduce
lookahead. It is knowingly conservative: a genuine same-session effect is shifted
a day later and therefore understated. Assuming intraday publication instead
would risk precisely the bias the live pipeline was built to eliminate.
"""

import datetime as dt

import pandas as pd

import database as db
from scraper import TICKERS

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
    "Stock_news/All_external.csv"
)

# The export is ordered by symbol, so the entire file must be scanned to reach
# every ticker in the universe. Chunking keeps memory flat while doing so.
CHUNK_ROWS = 200_000

# Inclusive publication-date bounds; None disables a bound.
START_DATE = "2018-01-01"
END_DATE = None

UNIVERSE = set(TICKERS)


def _publication_timestamp(raw):
    """Convert an FNSPID Date value to an aware UTC datetime, or None."""
    stamp = pd.to_datetime(raw, errors="coerce", format="mixed", utc=True)
    if pd.isna(stamp):
        return None

    stamp = stamp.to_pydatetime()
    if (stamp.hour, stamp.minute, stamp.second) == (0, 0, 0):
        # Date-only record: pinned to just before midnight exchange time so that
        # database.to_session_date routes it through its after-close branch and
        # into the following session.
        stamp = dt.datetime.combine(
            stamp.date(), dt.time(23, 59), tzinfo=db.EXCHANGE_TZ
        )
    return stamp.astimezone(dt.timezone.utc)


def backfill(start=START_DATE, end=END_DATE, url=FNSPID_URL):
    """Stream, filter and upsert historical headlines.

    Returns:
        The number of articles newly inserted.
    """
    print(f"Streaming FNSPID for {len(UNIVERSE)} tickers "
          f"({start or 'earliest'} .. {end or 'latest'})")
    print("The export is 5.7 GB and ordered by symbol, so a full scan is required.\n")

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    reader = pd.read_csv(
        url,
        chunksize=CHUNK_ROWS,
        dtype=str,
        usecols=["Date", "Article_title", "Stock_symbol", "Url"],
    )

    scanned = matched = new_articles = new_links = 0
    bounds = []

    with db.connect() as conn:
        db.init_db(conn)

        for index, chunk in enumerate(reader, start=1):
            scanned += len(chunk)
            chunk = chunk[chunk["Stock_symbol"].isin(UNIVERSE)]

            if not chunk.empty:
                stamps = pd.Series(
                    [_publication_timestamp(value) for value in chunk["Date"]],
                    index=chunk.index, dtype="object",
                )
                as_timestamp = pd.to_datetime(stamps, utc=True, errors="coerce")

                keep = as_timestamp.notna()
                if start_ts is not None:
                    keep &= as_timestamp >= start_ts
                if end_ts is not None:
                    keep &= as_timestamp <= end_ts

                chunk = chunk[keep]
                stamps = stamps[keep]
                as_timestamp = as_timestamp[keep]

            if not chunk.empty:
                records = [
                    {"ticker": symbol, "headline": title,
                     "published_at": stamp, "link": link}
                    for symbol, title, link, stamp in zip(
                        chunk["Stock_symbol"], chunk["Article_title"],
                        chunk["Url"], stamps)
                    if isinstance(title, str) and isinstance(link, str)
                ]

                if records:
                    stats = db.upsert_articles(conn, records)
                    matched += len(records)
                    new_articles += stats["articles_new"]
                    new_links += stats["links_new"]
                    bounds.extend([as_timestamp.min(), as_timestamp.max()])

                    # Committed as we go: a full scan runs long enough that
                    # losing all progress to a dropped connection would be
                    # costly, and the upsert makes restarting harmless.
                    conn.commit()

            if index % 10 == 0:
                print(f"  scanned {scanned:>11,}   kept {matched:>8,}   "
                      f"new {new_articles:>8,}")

        print(f"\n  scanned {scanned:,} rows")
        print(f"  matched {matched:,} in-universe headlines")
        print(f"  {new_articles:,} new articles, {new_links:,} new ticker links")
        if bounds:
            print(f"  publication range: {min(bounds)} .. {max(bounds)}")

        print("\nDatabase contents:")
        for table, count in db.table_counts(conn).items():
            print(f"  {table:<18} {count:>8} rows")

    return new_articles


if __name__ == "__main__":
    backfill()