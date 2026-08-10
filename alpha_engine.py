"""Pipeline stage 4: train and evaluate the directional classifier.

Reads daily_features, where calendar alignment, sentiment aggregation and label
construction have already happened. This module is concerned only with fitting
and evaluating.

Evaluation reports the majority-class baseline alongside model accuracy. In a
rising market the up-day share sits well above 50%, and an accuracy figure
quoted without that reference point is uninterpretable.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import database as db

FEATURES = ["mean_sentiment", "sum_sentiment", "headline_count", "volume"]
TEST_FRACTION = 0.2

# Below this many labelled rows, evaluation is dominated by sampling noise and
# is reported only to confirm the pipeline runs end to end.
CREDIBLE_SAMPLE = 300


def load_training_frame():
    """Load labelled rows from daily_features in chronological order."""
    with db.connect(read_only=True) as conn:
        return pd.read_sql_query(
            """
            SELECT ticker, session_date, mean_sentiment, sum_sentiment,
                   headline_count, close, volume, fwd_return, target
            FROM daily_features
            WHERE target IS NOT NULL
            ORDER BY session_date, ticker
            """,
            conn,
            parse_dates=["session_date"],
        )


def build_alpha_engine():
    """Fit the classifier on the stored feature matrix and report evaluation."""
    frame = load_training_frame()

    if frame.empty:
        print("No labelled rows. Run build_features.py first.")
        return

    up_share = frame["target"].mean()
    baseline = max(up_share, 1 - up_share)

    print(f"Loaded {len(frame)} labelled ticker-days "
          f"across {frame['session_date'].nunique()} sessions "
          f"and {frame['ticker'].nunique()} tickers.")
    print(f"  up-days {up_share:.1%}  ->  majority-class baseline {baseline:.1%}")

    if len(frame) < CREDIBLE_SAMPLE:
        print(f"\n  NOTE: fewer than {CREDIBLE_SAMPLE} labelled rows. The metrics")
        print("  below confirm the pipeline runs; they are not evidence of skill.")

    # Split on a session boundary rather than a row index. Rows from one trading
    # day share market-wide information, so splitting inside a day would leak it
    # across the boundary. A shuffled split would be worse still: it trains on
    # later sessions to predict earlier ones.
    sessions = frame["session_date"].drop_duplicates().sort_values()
    if len(sessions) < 3:
        print("\nNeed at least three distinct sessions to form a split.")
        return

    cutoff = sessions.iloc[int(len(sessions) * (1 - TEST_FRACTION))]
    train = frame[frame["session_date"] < cutoff]
    test = frame[frame["session_date"] >= cutoff]

    if train.empty or test.empty:
        print("\nSplit produced an empty side; more sessions are needed.")
        return

    print(f"\n  train {len(train):>4} rows, through {train['session_date'].max().date()}")
    print(f"  test  {len(test):>4} rows, from    {test['session_date'].min().date()}")

    model = XGBClassifier(eval_metric="logloss", max_depth=3, learning_rate=0.1)
    model.fit(train[FEATURES], train["target"])
    predictions = model.predict(test[FEATURES])

    accuracy = accuracy_score(test["target"], predictions)
    test_share = test["target"].mean()
    test_baseline = max(test_share, 1 - test_share)

    print("\n--- Evaluation ---")
    print(f"  model accuracy     {accuracy:.1%}")
    print(f"  baseline (majority){test_baseline:>8.1%}")
    print(f"  edge over baseline {accuracy - test_baseline:+.1%}")
    print("\n" + classification_report(test["target"], predictions, zero_division=0))

    print("Feature importance:")
    for name, weight in sorted(zip(FEATURES, model.feature_importances_),
                               key=lambda pair: -pair[1]):
        print(f"  {name:<16} {weight:.3f}")


if __name__ == "__main__":
    build_alpha_engine()