"""Import articles exported from a collection host.

Reads the JSON that export_articles.py writes, from stdin or a file, and feeds it
through the same upsert the scraper uses. Articles already present are recognised
by their canonical-URL hash and only refresh last_seen_at, so importing the same
window repeatedly is free and safe.

    ssh host 'cd nlp-alpha-engine && venv/bin/python export_articles.py' \\
        | python import_articles.py
"""

import argparse
import datetime as dt
import json
import sys

import database as db


def _to_timestamp(value):
    """Parse the ISO-8601 UTC string the exporter emits.

    database.parse_published expects RFC 2822, which is what RSS delivers, and
    would reject an ISO string. Converting to a datetime here is the cheaper fix:
    parse_published accepts one unchanged, so the ingestion path is shared rather
    than forked on input format.
    """
    text = str(value).strip().replace("Z", "+00:00")
    stamp = dt.datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp


def import_records(records):
    """Upsert exported records. Returns the counts upsert_articles reports."""
    prepared = []
    skipped = 0

    for record in records:
        try:
            prepared.append({
                "ticker": record["ticker"],
                "headline": record["headline"],
                "published_at": _to_timestamp(record["published_at"]),
                "link": record["link"],
            })
        except (KeyError, ValueError):
            # A malformed record is dropped rather than aborting the batch; the
            # count is reported so a systematic problem is visible.
            skipped += 1

    with db.connect() as conn:
        db.init_db(conn)
        stats = db.upsert_articles(conn, prepared)
        stats["total"] = db.table_counts(conn)["raw_news"]

    stats["skipped"] = skipped
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", nargs="?", help="JSON file; omit to read stdin")
    arguments = parser.parse_args()

    if arguments.path:
        with open(arguments.path, encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        records = json.load(sys.stdin)

    stats = import_records(records)

    print(f"received {len(records):,} records")
    print(f"  {stats['articles_seen']:,} unique articles, "
          f"{stats['articles_new']:,} new, {stats['links_new']:,} new ticker links")
    if stats["skipped"]:
        print(f"  {stats['skipped']:,} malformed records skipped")
    print(f"  {stats['total']:,} articles in the database")


if __name__ == "__main__":
    main()
