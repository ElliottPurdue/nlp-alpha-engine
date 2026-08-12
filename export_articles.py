"""Export recently collected articles as JSON, for transfer to another machine.

Runs on the collection host. Emits one record per (article, ticker) pair -- the
same shape the scraper produces and upsert_articles consumes -- so the receiving
side reconstructs both raw_news and its junction table through the ordinary
ingestion path rather than a special-cased merge.

Exports a rolling window rather than tracking a watermark. Re-exporting articles
the receiver already holds costs nothing, because the upsert is idempotent, and a
window is self-healing: a sync missed for three days is repaired by the next one
instead of leaving a permanent hole that watermark bookkeeping would hide.

Writes to stdout so it can be piped straight over ssh:

    ssh host 'cd nlp-alpha-engine && venv/bin/python export_articles.py' \\
        | python import_articles.py
"""

import argparse
import json
import sys

import database as db

DEFAULT_WINDOW_DAYS = 7

# A collection host gathers on the order of a thousand articles a day, so a
# week's window is a few thousand records and a couple of megabytes. The cap
# exists because first_seen_at records when *this* database first saw a row, and
# a bulk import stamps its entire contents with the same moment -- run against a
# database that has just ingested a historical corpus, an innocent-looking
# three-day window returned 310,000 records and 76 MB. That is fine to hold on
# disk and wrong to push through an ssh pipe.
DEFAULT_MAX_RECORDS = 50_000


def export(days=DEFAULT_WINDOW_DAYS, max_records=DEFAULT_MAX_RECORDS):
    """Records for articles first seen within the last `days`.

    Returns (records, truncated). `truncated` is true if the cap was reached, so
    the caller can report it rather than silently sending a partial window.
    """
    with db.connect(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT t.ticker, n.headline, n.published_at, n.url
            FROM raw_news n
            JOIN news_tickers t USING (article_id)
            WHERE n.first_seen_at >= datetime('now', ?)
            ORDER BY n.first_seen_at DESC
            LIMIT ?
            """,
            (f"-{int(days)} days", int(max_records) + 1),
        ).fetchall()

    truncated = len(rows) > max_records
    rows = rows[:max_records]

    return ([{"ticker": r["ticker"], "headline": r["headline"],
              "published_at": r["published_at"], "link": r["url"]}
             for r in rows], truncated)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS,
                        help="rolling window to export")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS,
                        help="safety cap on the export size")
    arguments = parser.parse_args()

    records, truncated = export(arguments.days, arguments.max_records)
    json.dump(records, sys.stdout)

    # Progress goes to stderr so stdout stays a clean JSON stream for the pipe.
    print(f"exported {len(records):,} ticker-article records "
          f"from the last {arguments.days} days", file=sys.stderr)
    if truncated:
        print(f"WARNING: hit the {arguments.max_records:,} record cap; the "
              f"newest records were sent and older ones in the window were not. "
              f"Re-run with a shorter --days, or raise --max-records if this is "
              f"genuinely a large backlog.", file=sys.stderr)


if __name__ == "__main__":
    main()
