"""Backfill historical headlines from the FNSPID dataset.

Streams an FNSPID news export from HuggingFace in chunks, filters to the universe
and date range, and writes through the same upsert the live scraper uses. Nothing
is stored on disk.

Two exports are available and they are very different objects:

    benzinga  All_external.csv, 5.7 GB, roughly 2009-2020, headlines only.
    nasdaq    nasdaq_exteral_data.csv, 23.2 GB, roughly 2007-2023, with full
              article bodies at a median of 4,750 characters.

Both are ordered by symbol, so the whole file must be scanned to reach every
ticker regardless of the date filter. The bodies in the nasdaq export make its
records about fifteen times larger, which is why it reads in much smaller chunks.

One difference from live collection is deliberate. FNSPID records are largely
date-only -- 96.8% in the benzinga export, 99.6% in the nasdaq one -- so whether an
article preceded or followed the close is unknowable. Those records are attributed
to the FOLLOWING session, the only assumption that cannot leak. It understates any
genuine same-session effect, which is the right direction to be wrong in.

Usage:
    python backfill_news.py nasdaq
    python backfill_news.py benzinga --start 2018-01-01
"""

import argparse
import datetime as dt

import pandas as pd

import database as db
from scraper import TICKERS

_BASE = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/"

# Columns are read by name and are identical across both exports; the nasdaq file
# simply carries eight more that are not needed here. Selecting a subset does not
# avoid tokenizing the article bodies, but it does keep them out of memory.
FIELDS = ["Date", "Article_title", "Stock_symbol", "Url"]

SOURCES = {
    "benzinga": {
        "file": "All_external.csv",
        # Headline-only records are small, so large chunks are cheap.
        "chunk_rows": 200_000,
        "size_gb": 5.7,
        "default_start": "2018-01-01",
    },
    "nasdaq": {
        "file": "nasdaq_exteral_data.csv",
        # Records average ~6.5 KB once article bodies are included. At 200k rows a
        # chunk would be well over a gigabyte of raw text before any filtering.
        "chunk_rows": 20_000,
        "size_gb": 23.2,
        "default_start": None,
    },
}

# Progress is reported every this many chunks.
PROGRESS_EVERY = 10

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


def backfill(source="nasdaq", start=None, end=None):
    """Stream, filter and upsert historical headlines. Returns articles inserted."""
    config = SOURCES[source]
    url = _BASE + config["file"]
    start = config["default_start"] if start is None else start

    print(f"Streaming {config['file']} ({config['size_gb']} GB) for "
          f"{len(UNIVERSE)} tickers")
    print(f"  date filter: {start or 'earliest'} .. {end or 'latest'}")
    print("  the export is ordered by symbol, so a full scan is required\n")

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    reader = pd.read_csv(
        url,
        chunksize=config["chunk_rows"],
        dtype=str,
        usecols=FIELDS,
        # Article bodies contain unescaped quotes often enough that a strict parse
        # would abort partway through a 23 GB scan. Skipping the handful of
        # malformed records costs less than restarting.
        on_bad_lines="skip",
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

            if index % PROGRESS_EVERY == 0:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="?", default="nasdaq",
                        choices=sorted(SOURCES),
                        help="which FNSPID export to stream")
    parser.add_argument("--start", help="earliest publication date, YYYY-MM-DD")
    parser.add_argument("--end", help="latest publication date, YYYY-MM-DD")
    arguments = parser.parse_args()
    backfill(arguments.source, arguments.start, arguments.end)


if __name__ == "__main__":
    main()
