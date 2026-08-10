"""Walk-forward evaluation of the cross-sectional classifier.

A single chronological split yields one verdict from one market regime. Because
the historical corpus ends in June 2020, that split always places the COVID crash
in the test set, so no feature change can be meaningfully assessed against it.

This module refits at a fixed cadence and evaluates each successive out-of-sample
window, producing a prediction for every session from a model that never saw it.
That is both a fairer evaluation and precisely the input a backtest requires.

Reported alongside accuracy is the information coefficient: the per-session rank
correlation between predicted score and realized excess return. For a
cross-sectional signal the IC is the more informative statistic, because a
strategy profits from ranking names correctly rather than from classifying each
one in isolation.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from alpha_engine import FEATURE_SETS, TARGET, load_training_frame

# Sessions fitted before the first evaluation window; roughly one trading year.
INITIAL_TRAIN_SESSIONS = 250

# Sessions between refits; roughly one month. Frequent enough to adapt, cheap
# enough that the whole pass runs in seconds.
STEP_SESSIONS = 21

# A rank correlation computed on a handful of names is dominated by which names
# happened to have news that day, so thin sessions are excluded from the IC.
MIN_NAMES_FOR_IC = 5


def _fit_predict(train, test, features, target):
    """Fit on the training window and return P(outperform) for the test window."""
    model = XGBClassifier(eval_metric="logloss", max_depth=3, learning_rate=0.1)
    model.fit(train[features], train[target])
    return model.predict_proba(test[features])[:, 1]


def walk_forward(frame=None, features=None, target=TARGET,
                 initial=INITIAL_TRAIN_SESSIONS, step=STEP_SESSIONS):
    """Generate out-of-sample predictions across the full history.

    The training window expands rather than rolls: each refit uses all history
    available at that point, which is what a live system would have had.

    Args:
        frame: Feature matrix; loaded for `target` if omitted.
        features: Feature column names; defaults to the level set.
        target: Label column to fit against.

    Returns:
        (predictions, windows). `predictions` is every evaluated row with a
        `score` column added; `windows` summarises one row per refit.
    """
    features = FEATURE_SETS["level"] if features is None else features
    frame = load_training_frame(target) if frame is None else frame

    # Rows lacking a feature value cannot be scored. Surprise columns are NULL
    # until a ticker has accumulated enough history for a baseline, so this
    # legitimately trims the early part of the sample for those feature sets.
    frame = frame.dropna(subset=list(features) + [target]).reset_index(drop=True)

    sessions = (
        frame["session_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    if len(sessions) <= initial:
        raise SystemExit(
            f"Need more than {initial} sessions to walk forward; have {len(sessions)}."
        )

    collected, windows = [], []

    for start in range(initial, len(sessions), step):
        train_end = sessions.iloc[start]
        test_end = sessions.iloc[min(start + step, len(sessions)) - 1]

        train = frame[frame["session_date"] < train_end]
        test = frame[(frame["session_date"] >= train_end)
                     & (frame["session_date"] <= test_end)]

        if train.empty or test.empty:
            continue

        scored = test.copy()
        scored["score"] = _fit_predict(train, test, features, target)
        collected.append(scored)

        predicted = (scored["score"] > 0.5).astype(int)
        share = scored[target].mean()
        windows.append({
            "from": train_end.date(),
            "to": test_end.date(),
            "train": len(train),
            "test": len(test),
            "accuracy": (predicted == scored[target]).mean(),
            "baseline": max(share, 1 - share),
        })

    return pd.concat(collected, ignore_index=True), pd.DataFrame(windows)


def information_coefficient(predictions, excess_column="excess_return",
                            stride=1, min_names=MIN_NAMES_FOR_IC):
    """Per-session Spearman correlation between score and realized excess return.

    Args:
        predictions: Output of walk_forward.
        excess_column: Realized excess return to correlate against.
        stride: Keep only every nth session. For a multi-day horizon consecutive
            observations share most of their return window, so their ICs are
            strongly autocorrelated and a naive t-statistic would be inflated;
            striding by the horizon length restores independence at the cost of
            sample size.

    Returns:
        A Series indexed by session date. Spearman rather than Pearson because
        only the ordering of names matters to a long/short book, and ranks are
        insensitive to the fat tails of daily returns.
    """
    values = {}
    for session, group in predictions.groupby("session_date"):
        if len(group) < min_names:
            continue
        correlation = group["score"].corr(group[excess_column], method="spearman")
        if pd.notna(correlation):
            values[session] = correlation

    series = pd.Series(values).sort_index()
    return series.iloc[::stride] if stride > 1 else series


def summarize(predictions, target=TARGET, excess_column="excess_return", stride=1):
    """Return pooled accuracy and IC statistics for a set of predictions."""
    predicted = (predictions["score"] > 0.5).astype(int)
    share = predictions[target].mean()

    ic = information_coefficient(predictions, excess_column, stride)
    ic_std = ic.std()
    # Standard t-statistic on the mean of per-session ICs. |t| above roughly 2 is
    # the conventional threshold for a signal distinguishable from noise.
    t_stat = ic.mean() / ic_std * np.sqrt(len(ic)) if ic_std else 0.0

    return {
        "rows": len(predictions),
        "sessions": predictions["session_date"].nunique(),
        "accuracy": (predicted == predictions[target]).mean(),
        "baseline": max(share, 1 - share),
        "ic_sessions": len(ic),
        "mean_ic": ic.mean(),
        "ic_std": ic_std,
        "t_stat": t_stat,
        "ic_hit_rate": (ic > 0).mean(),
    }


def run():
    """Evaluate the default configuration and report per-window statistics."""
    predictions, windows = walk_forward()

    print(f"=== {len(windows)} walk-forward windows ===")
    print(f"  {'from':<12}{'to':<12}{'train':>8}{'test':>7}"
          f"{'acc':>8}{'base':>8}{'edge':>8}")
    for row in windows.itertuples():
        edge = row.accuracy - row.baseline
        print(f"  {str(row._1):<12}{str(row.to):<12}{row.train:>8,}{row.test:>7,}"
              f"{row.accuracy:>8.1%}{row.baseline:>8.1%}{edge:>+8.1%}")

    stats = summarize(predictions)

    print("\n=== pooled out-of-sample ===")
    print(f"  rows        {stats['rows']:>10,}")
    print(f"  sessions    {stats['sessions']:>10,}")
    print(f"  accuracy    {stats['accuracy']:>10.1%}")
    print(f"  baseline    {stats['baseline']:>10.1%}")
    print(f"  edge        {stats['accuracy'] - stats['baseline']:>+10.1%}")

    print("\n=== information coefficient ===")
    print(f"  sessions    {stats['ic_sessions']:>10,}")
    print(f"  mean IC     {stats['mean_ic']:>+10.4f}")
    print(f"  IC std      {stats['ic_std']:>10.4f}")
    print(f"  t-statistic {stats['t_stat']:>+10.2f}")
    print(f"  hit rate    {stats['ic_hit_rate']:>10.1%}  (sessions with positive IC)")

    verdict = ("no statistically detectable ranking skill"
               if abs(stats["t_stat"]) < 2 else "signal distinguishable from noise")
    print(f"\n  |t| {'<' if abs(stats['t_stat']) < 2 else '>='} 2  ->  {verdict}")

    return predictions


if __name__ == "__main__":
    run()
