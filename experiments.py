"""Pre-registered hypothesis tests.

Both hypotheses below were stated before any of these results were computed, and
every configuration is reported regardless of outcome. That distinction matters:
running four tests and reporting four is an experiment, while running forty and
reporting the best is a story about noise.

    H0  Level features over a one-day horizon. The existing configuration,
        included as the reference point rather than as a hypothesis.

    H1  Surprise beats level. Cross-sectional ranks describe how a stock's news
        compares with its peers; they cannot express how it compares with the
        stock's own normal. Published work associates returns with abnormal
        attention and tone rather than with their levels, and the current feature
        set has no way to represent that.

    H2  One day is too short. Daily returns are dominated by microstructure
        noise. A five-day horizon gives any genuine effect room to surface.

Because four configurations are tested, the conventional |t| > 2 threshold is too
permissive. A Bonferroni correction for four tests puts the 5% threshold at
roughly |t| > 2.5, which is applied when reporting verdicts.

Overlapping observations are handled by striding the five-day IC series by five
sessions: consecutive five-day returns share four of their five days, so their
per-session ICs are strongly autocorrelated and an uncorrected t-statistic would
overstate significance considerably.
"""

from alpha_engine import FEATURE_SETS
from walkforward import summarize, walk_forward

# (label, feature set, target column, realized excess column, IC stride)
HYPOTHESES = [
    ("H0  level / 1-day", "level", "target_relative", "excess_return", 1),
    ("H1  surprise / 1-day", "surprise", "target_relative", "excess_return", 1),
    ("H2  level / 5-day", "level", "target_relative_5d", "excess_return_5d", 5),
    ("H1+H2  surprise / 5-day", "surprise", "target_relative_5d",
     "excess_return_5d", 5),
    ("aux  combined / 1-day", "combined", "target_relative", "excess_return", 1),
]

# Bonferroni-adjusted two-sided 5% threshold for the number of tests run.
SIGNIFICANCE_T = 2.5


def run():
    """Evaluate every pre-registered configuration and report all outcomes."""
    results = []

    for label, feature_set, target, excess_column, stride in HYPOTHESES:
        features = FEATURE_SETS[feature_set]
        print(f"running {label} ...")
        predictions, _ = walk_forward(features=features, target=target)
        stats = summarize(predictions, target=target,
                          excess_column=excess_column, stride=stride)
        stats["label"] = label
        results.append(stats)

    print(f"\n{'=' * 78}")
    print(f"  {'configuration':<26}{'rows':>7}{'acc':>8}{'base':>8}"
          f"{'edge':>8}{'mean IC':>10}{'IC n':>6}{'t':>7}")
    print(f"{'-' * 78}")
    for stats in results:
        print(f"  {stats['label']:<26}{stats['rows']:>7,}"
              f"{stats['accuracy']:>8.1%}{stats['baseline']:>8.1%}"
              f"{stats['accuracy'] - stats['baseline']:>+8.1%}"
              f"{stats['mean_ic']:>+10.4f}{stats['ic_sessions']:>6}"
              f"{stats['t_stat']:>+7.2f}")
    print(f"{'=' * 78}")

    significant = [s for s in results if abs(s["t_stat"]) >= SIGNIFICANCE_T]

    print(f"\nSignificance threshold |t| >= {SIGNIFICANCE_T} "
          f"(Bonferroni-corrected for {len(HYPOTHESES)} tests)")
    if significant:
        for stats in significant:
            direction = "positive" if stats["mean_ic"] > 0 else "NEGATIVE"
            print(f"  {stats['label']}: mean IC {stats['mean_ic']:+.4f} "
                  f"(t = {stats['t_stat']:+.2f}) -> {direction} ranking skill")
    else:
        print("  No configuration reaches it. The null result stands, and it now")
        print("  stands against a genuine attempt to overturn it rather than by")
        print("  default.")

    # The smallest true effect this sample could have detected. Reporting it
    # distinguishes "no signal exists" from "no signal large enough to see here",
    # which are very different claims.
    print("\nMinimum detectable mean IC per configuration:")
    for stats in results:
        if stats["ic_sessions"]:
            floor = SIGNIFICANCE_T * stats["ic_std"] / stats["ic_sessions"] ** 0.5
            print(f"  {stats['label']:<26}{floor:>8.4f}  "
                  f"({stats['ic_sessions']} independent sessions)")

    return results


if __name__ == "__main__":
    run()
