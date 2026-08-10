"""Pipeline stage 2: score unscored headlines with FinBERT.

Only articles carrying no score for the configured model are read, so this stage
costs time proportional to new input rather than to accumulated history. That
matters because model inference dominates pipeline runtime.

Because headline text is persisted at collection time, scoring is decoupled from
collection entirely: a backlog of any size can be cleared in a single pass,
whenever it is convenient to run.
"""

from transformers import pipeline

import database as db

BATCH_SIZE = 32

# Scores are committed in groups of roughly this size. A backfill pass runs for
# around an hour, so buffering everything to the end would mean an interruption
# discarded all of it. Committing periodically caps the loss at one group and
# makes the pass stoppable and resumable.
COMMIT_EVERY = 320
# FinBERT's context window. Headlines fall well short of it, but truncating
# guards against an unusually long title raising mid-batch.
MAX_TOKENS = 512

# Progress is reported every this many batches. Frequent enough to show the run
# is alive, sparse enough not to flood a log file.
PROGRESS_EVERY = 10


def analyze_sentiment(model_name=db.FINBERT_MODEL):
    """Score every unscored article and persist the results.

    Args:
        model_name: HuggingFace model identifier, recorded with each score so
            that output from different models can coexist.

    Returns:
        The number of headlines scored.
    """
    with db.connect() as conn:
        db.init_db(conn)
        pending = db.fetch_unscored_articles(conn, model_name)

        if not pending:
            print("No unscored headlines; the database is up to date.")
            return 0

        print(f"Found {len(pending)} unscored headlines.")
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
                # Committed mid-pass so an interrupted run keeps its work; the
                # anti-join then resumes from exactly where it stopped.
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
