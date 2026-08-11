"""Walk-forward evaluation.

A single split gives one verdict from one regime, and since the corpus ends in June
2020 that split always lands on the COVID crash. Refitting monthly and scoring each
following window instead gives eighteen windows across calm markets, the Q4 2018
selloff, the crash and the recovery. It also produces exactly what a backtest needs.

Accuracy is reported next to the information coefficient, the per-session rank
correlation between score and realized excess return. The IC is the more useful
number here: a long/short book profits from ranking names correctly, not from
classifying each one.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from alpha_engine import FEATURE_SETS, TARGET, load_training_frame

# Sessions fitted before the first evaluation window; about a trading year.
INITIAL_TRAIN_SESSIONS = 250

# Sessions between refits. Roughly monthly: often enough to adapt, cheap enough
# that a full pass takes seconds.
STEP_SESSIONS = 21

# A rank correlation over a handful of names says more about which names had news
# than about the signal, so thin sessions are left out of the IC.
MIN_NAMES_FOR_IC = 5


def _fit_predict(train, test, features, target):
    model = XGBClassifier(eval_metric="logloss", max_depth=3, learning_rate=0.1)
    model.fit(train[features], train[target])
    return model.predict_proba(test[features])[:, 1]


def walk_forward(frame=None, features=None, target=TARGET,
                 initial=INITIAL_TRAIN_SESSIONS, step=STEP_SESSIONS):
    """Score every session from a model that never saw it.

    The training window expands rather than rolls, so each refit uses all the
    history a live system would have had.

    Returns (predictions, windows): the evaluated rows with a `score` column, and
    one summary row per refit.
    """
    features = FEATURE_SETS["level"] if features is None else features
    frame = load_training_frame(target) if frame is None else frame

    # Surprise columns are NULL until a ticker has enough history for a baseline,
    # so this legitimately trims the start of the sample for those feature sets.
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

    Spearman because only the ordering matters to a long/short book, and ranks
    shrug off the fat tails of daily returns.

    `stride` keeps every nth session. For a multi-day horizon, consecutive
    observations share most of their return window, so their ICs are autocorrelated
    and a naive t-statistic is inflated; striding by the horizon restores
    independence at the cost of sample size.
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
    """Pooled accuracy and IC statistics for one set of predictions."""
    predicted = (predictions["score"] > 0.5).astype(int)
    share = predictions[target].mean()

    ic = information_coefficient(predictions, excess_column, stride)
    ic_std = ic.std()
    # t on the mean of per-session ICs. |t| above about 2 is the usual bar for a
    # signal distinguishable from noise.
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
