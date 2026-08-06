"""Import the pre-database CSV extracts into SQLite.

Retained after the initial migration rather than deleted: the database file is
not version controlled, so this script and the CSV extracts together form the
recovery path for the earliest headlines, which have since rotated out of the
source feed and cannot be scraped again.

The import is idempotent and may be re-run safely.
"""

import pandas as pd

import database as db


def migrate():
    """Load both CSV extracts into the database, then report table sizes."""
    with db.connect() as conn:
        db.init_db(conn)

        raw_path = db.PROJECT_DIR / "raw_headlines.csv"
        scored_path = db.PROJECT_DIR / "scored_headlines.csv"

        # The scored extract is normally a superset of the raw one, but both are
        # imported in case the scraper ran more recently than the classifier.
        for path in (raw_path, scored_path):
            if not path.exists():
                print(f"Skipping {path.name} (not found)")
                continue
            frame = pd.read_csv(path)
            # to_dict("records") yields the same shape the scraper emits, so one
            # upsert implementation serves both migration and live collection.
            stats = db.upsert_articles(conn, frame.to_dict("records"))
            print(f"{path.name}: {stats['articles_seen']} articles seen, "
                  f"{stats['articles_new']} new, {stats['links_new']} new ticker links")

        if scored_path.exists():
            scored = pd.read_csv(scored_path)
            scored = scored.dropna(subset=["sentiment_label", "sentiment_score"])

            # The extracts predate the database and carry no article_id, so each
            # link is re-fingerprinted to locate the row inserted above.
            ids = db.article_ids_by_fingerprint(
                conn, (db.url_fingerprint(u) for u in scored["link"])
            )
            triples = [
                (ids[db.url_fingerprint(row.link)], row.sentiment_label, row.sentiment_score)
                for row in scored.itertuples()
                if db.url_fingerprint(row.link) in ids
            ]
            n = db.upsert_sentiment(conn, triples)
            print(f"scored_headlines.csv: {n} sentiment scores upserted")

        print("\nDatabase contents:")
        for table, count in db.table_counts(conn).items():
            print(f"  {table:<18} {count:>6} rows")


if __name__ == "__main__":
    migrate()
