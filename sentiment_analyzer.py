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

        scored = []
        for batch_index, start in enumerate(range(0, len(pending), BATCH_SIZE)):
            batch = pending[start:start + BATCH_SIZE]
            results = classifier(
                [row["headline"] for row in batch],
                truncation=True,
                max_length=MAX_TOKENS,
            )
            scored.extend(
                (row["article_id"], res["label"], res["score"])
                for row, res in zip(batch, results)
            )
            if batch_index % PROGRESS_EVERY == 0 or len(scored) == len(pending):
                print(f"  scored {len(scored)}/{len(pending)}")

        # A single upsert inside the caller's transaction: either every score
        # from this pass lands or none does, so an interrupted run leaves no
        # partially scored batch behind.
        count = db.upsert_sentiment(conn, scored, model_name)

        distribution = {}
        for _, label, _ in scored:
            key = label.lower()
            distribution[key] = distribution.get(key, 0) + 1

    print(f"\nScored {count} headlines:")
    for label in ("positive", "neutral", "negative"):
        share = distribution.get(label, 0) / count if count else 0
        print(f"  {label:<9} {distribution.get(label, 0):>5}  ({share:.1%})")

    return count


if __name__ == "__main__":
    analyze_sentiment()
