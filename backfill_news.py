"""Backfill historical headlines from the FNSPID dataset.

Streams FNSPID's headline export from HuggingFace in chunks, filters to the
universe and date range, and writes through the same upsert the live scraper uses.
Nothing is stored on disk.

One difference from live collection is deliberate. FNSPID records are largely
date-only, so whether an article preceded or followed the close is unknowable.
Those records are attributed to the FOLLOWING session, the only assumption that
cannot leak. It understates any genuine same-session effect, which is the right
direction to be wrong in.
"""

import datetime as dt

import pandas as pd

import database as db
from scraper import TICKERS

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
    "Stock_news/All_external.csv"
)

# The export is ordered by symbol, so the whole file must be scanned to reach every
# ticker. Chunking keeps memory flat while doing it.
CHUNK_ROWS = 200_000

# Inclusive publication-date bounds; None disables a bound.
START_DATE = "2018-01-01"
END_DATE = None

UNIVERSE = set(TICKERS)


def _publication_timestamp(raw):
    """FNSPID Date value to aware UTC, or None if unparseable."""
    stamp = pd.to_datetime(raw, errors="coerce", format="mixed", utc=True)
    if pd.isna(stamp):
        return None

    stamp = stamp.to_pydatetime()
    if (stamp.hour, stamp.minute, stamp.second) == (0, 0, 0):
        # Date-only. Pinned just before midnight exchange time so that
        # to_session_date takes its after-close branch.
        stamp = dt.datetime.combine(
            stamp.date(), dt.time(23, 59), tzinfo=db.EXCHANGE_TZ
        )
    return stamp.astimezone(dt.timezone.utc)


def backfill(start=START_DATE, end=END_DATE, url=FNSPID_URL):
    """Stream, filter and upsert historical headlines. Returns articles inserted."""
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

                    # Committed as we go. A full scan runs long enough that losing
                    # everything to a dropped connection would hurt, and the upsert
                    # makes restarting harmless.
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
