"""Stage 4: train and evaluate the cross-sectional classifier.

The question is relative, not directional: within a session, which names beat the
cross-section? Over half the variance of an individual daily return is the market
moving, and company news cannot forecast that, so asking for absolute direction
buries whatever stock-selection signal exists under noise the features could never
explain.

Accuracy is always reported next to the majority-class baseline. On its own it
means nothing.

For a fair evaluation use walkforward.py; the single split here permanently lands
its test window on the COVID crash, because the corpus ends in June 2020.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import database as db

# "level" compares a stock to its peers that day; "surprise" compares it to its own
# recent history. Different questions, kept separable so they can be compared.
FEATURE_SETS = {
    "level": ["sentiment_rank", "headline_rank", "volume_rank", "mean_sentiment"],
    "surprise": ["sentiment_surprise_rank", "attention_surprise_rank",
                 "sentiment_surprise", "attention_surprise"],
    "combined": ["sentiment_rank", "headline_rank", "volume_rank",
                 "sentiment_surprise_rank", "attention_surprise_rank"],
}

FEATURES = FEATURE_SETS["level"]
TARGET = "target_relative"

TEST_FRACTION = 0.2

# The backfill ends mid-2020 and live collection began in 2026. Fitting across that
# gap trains on one era and tests on another, so development uses the contiguous
# historical block and the live period is held back as an untouched forward test.
MODEL_PERIOD_END = "2020-06-30"

# Below this many rows, evaluation is sampling noise and is reported only to show
# the pipeline runs.
CREDIBLE_SAMPLE = 300


def load_training_frame(target=TARGET):
    """Labelled rows from the contiguous historical block, chronologically ordered.

    Rows lacking `target` are dropped. The five-day target is NULL for the last few
    sessions of each ticker, so the usable sample differs by horizon.
    """
    with db.connect(read_only=True) as conn:
        frame = pd.read_sql_query(
            """
            SELECT ticker, session_date, mean_sentiment, sum_sentiment,
                   headline_count, close, volume,
                   sentiment_rank, headline_rank, volume_rank,
                   sentiment_surprise, attention_surprise,
                   sentiment_surprise_rank, attention_surprise_rank,
                   fwd_return, excess_return, target, target_relative,
                   fwd_return_5d, excess_return_5d, target_relative_5d
            FROM daily_features
            WHERE session_date <= ?
            ORDER BY session_date, ticker
            """,
            conn,
            params=(MODEL_PERIOD_END,),
            parse_dates=["session_date"],
        )
    return frame[frame[target].notna()].reset_index(drop=True)


def build_alpha_engine():
    """Fit on one chronological split and report accuracy against the baseline."""
    frame = load_training_frame()

    if frame.empty:
        print("No labelled rows. Run build_features.py first.")
        return

    outperform_share = frame[TARGET].mean()
    baseline = max(outperform_share, 1 - outperform_share)

    print(f"Loaded {len(frame):,} labelled ticker-days "
          f"across {frame['session_date'].nunique()} sessions "
          f"and {frame['ticker'].nunique()} tickers "
          f"(through {MODEL_PERIOD_END}).")
    print(f"  outperformers {outperform_share:.1%}  ->  "
          f"majority-class baseline {baseline:.1%}")

    if len(frame) < CREDIBLE_SAMPLE:
        print(f"\n  NOTE: fewer than {CREDIBLE_SAMPLE} labelled rows. The metrics")
        print("  below confirm the pipeline runs; they are not evidence of skill.")

    # Split on a session boundary, picked by cumulative rows rather than session
    # count: session sizes differ by more than an order of magnitude while
    # collection ramps up. Splitting inside a session would also leak that day's
    # market-wide move across the boundary.
    cumulative = frame.groupby("session_date").size().sort_index().cumsum()
    eligible = cumulative[cumulative <= len(frame) * (1 - TEST_FRACTION)]

    if eligible.empty or len(eligible) == len(cumulative):
        print("\nSessions are too unevenly sized to form a clean split.")
        return

    cutoff = cumulative.index[len(eligible)]
    train = frame[frame["session_date"] < cutoff]
    test = frame[frame["session_date"] >= cutoff]

    print(f"\n  train {len(train):>6,} rows, through {train['session_date'].max().date()}")
    print(f"  test  {len(test):>6,} rows, from    {test['session_date'].min().date()}")

    model = XGBClassifier(eval_metric="logloss", max_depth=3, learning_rate=0.1)
    model.fit(train[FEATURES], train[TARGET])
    predictions = model.predict(test[FEATURES])

    accuracy = accuracy_score(test[TARGET], predictions)
    test_share = test[TARGET].mean()
    test_baseline = max(test_share, 1 - test_share)

    print("\n--- Evaluation ---")
    print(f"  model accuracy      {accuracy:>7.1%}")
    print(f"  baseline (majority) {test_baseline:>7.1%}")
    print(f"  edge over baseline  {accuracy - test_baseline:>+7.1%}")
    print("\n" + classification_report(test[TARGET], predictions, zero_division=0))

    print("Feature importance:")
    for name, weight in sorted(zip(FEATURES, model.feature_importances_),
                               key=lambda pair: -pair[1]):
        print(f"  {name:<16} {weight:.3f}")


if __name__ == "__main__":
    build_alpha_engine()
