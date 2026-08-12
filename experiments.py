"""Pre-registered hypothesis tests.

Both hypotheses were written down before any of these numbers existed, and every
configuration is reported whatever it shows. Running four tests and reporting four
is an experiment; running forty and reporting the best is a story about noise.

    H1  Surprise beats level. Cross-sectional ranks say how a stock's news compares
        with its peers but cannot say how it compares with the stock's own normal,
        and abnormal attention and tone are what the literature ties to returns.

    H2  One day is too short. Daily returns are mostly microstructure noise; five
        days gives a real effect room to show up.

With five configurations the usual |t| > 2 bar is too loose, so verdicts use an
exact two-sided Bonferroni critical value, 2.576 at this count.

A result that only clears the bar on the least powerful configuration, at an effect
size equal to that configuration's minimum detectable effect, is reported as what
it is: at the edge of what the sample can distinguish from noise.
"""

from scipy import stats

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

# Two-sided Bonferroni critical value, derived rather than rounded. At five tests
# this is 2.576, and the difference from a rounded 2.5 is not academic: a result at
# t = 2.54 reads as significant against one and not against the other.
FAMILY_ALPHA = 0.05
SIGNIFICANCE_T = stats.norm.ppf(1 - FAMILY_ALPHA / (2 * len(HYPOTHESES)))


def run():
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

    # Smallest true effect this sample could have caught. "No signal exists" and
    # "no signal large enough to see here" are very different claims.
    print("\nMinimum detectable mean IC per configuration:")
    for stats in results:
        if stats["ic_sessions"]:
            floor = SIGNIFICANCE_T * stats["ic_std"] / stats["ic_sessions"] ** 0.5
            print(f"  {stats['label']:<26}{floor:>8.4f}  "
                  f"({stats['ic_sessions']} independent sessions)")

    return results


if __name__ == "__main__":
    run()
