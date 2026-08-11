"""Stage 2: score unscored headlines with FinBERT.

Only articles with no score for this model are read, so cost scales with new input
rather than with total history. Because headline text is stored at collection time,
scoring is fully decoupled from collection: a backlog of any size can be cleared
whenever it is convenient.
"""

from transformers import pipeline

import database as db

BATCH_SIZE = 32

# Committed in groups. A backfill pass runs about an hour, so holding everything
# until the end would mean an interruption discarded all of it.
COMMIT_EVERY = 320

# FinBERT's context window. Headlines are far shorter, but truncation stops one
# unusually long title from raising mid-batch.
MAX_TOKENS = 512


def analyze_sentiment(model_name=db.FINBERT_MODEL):
    """Score everything unscored and return how many headlines were processed."""
    with db.connect() as conn:
        db.init_db(conn)
        pending = db.fetch_unscored_articles(conn, model_name)

        if not pending:
            print("No unscored headlines; the database is up to date.")
            return 0

        print(f"Found {len(pending):,} unscored headlines.")
        print(f"Loading {model_name} (the first run downloads the weights)...")
        classifier = pipeline("sentiment-analysis", model=model_name)

        distribution = {}
        buffer = []
        completed = 0

        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            results = classifier(
                [row["headline"] for row in batch],
                truncation=True,
                max_length=MAX_TOKENS,
            )

            for row, result in zip(batch, results):
                buffer.append((row["article_id"], result["label"], result["score"]))
                label = result["label"].lower()
                distribution[label] = distribution.get(label, 0) + 1

            if len(buffer) >= COMMIT_EVERY or start + BATCH_SIZE >= len(pending):
                db.upsert_sentiment(conn, buffer, model_name)
                # Committed mid-pass, so an interrupted run keeps its work and the
                # anti-join resumes from where it stopped.
                conn.commit()
                completed += len(buffer)
                buffer = []
                print(f"  scored {completed:,}/{len(pending):,}")

    print(f"\nScored {completed:,} headlines:")
    for label in ("positive", "neutral", "negative"):
        share = distribution.get(label, 0) / completed if completed else 0
        print(f"  {label:<9} {distribution.get(label, 0):>7,}  ({share:.1%})")

    return completed


if __name__ == "__main__":
    analyze_sentiment()
