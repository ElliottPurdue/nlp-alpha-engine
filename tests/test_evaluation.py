"""Walk-forward evaluation.

The split is what makes every reported number out-of-sample. If a fold ever fitted
on a session it later scored, results would improve and nothing would look wrong.
This asserts the boundary directly by recording what each fold actually saw.

Also covered: the stride that corrects for overlapping multi-day returns, which is
the difference between a defensible t-statistic and an inflated one.
"""

import unittest

import numpy as np
import pandas as pd

import walkforward as wf


def synthetic_panel(sessions=400, names=12, seed=0):
    """A panel with the columns walk_forward needs and no signal in it."""
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=sessions)
    rows = []
    for date in dates:
        for name in range(names):
            rows.append({
                "ticker": f"T{name:02d}",
                "session_date": date,
                "sentiment_rank": generator.random(),
                "headline_rank": generator.random(),
                "volume_rank": generator.random(),
                "mean_sentiment": generator.normal(0, 0.3),
                "excess_return": generator.normal(0, 0.02),
                "target_relative": generator.integers(0, 2),
            })
    return pd.DataFrame(rows)


class NoLeakage(unittest.TestCase):
    def test_every_fold_trains_strictly_before_it_predicts(self):
        panel = synthetic_panel()
        observed = []

        original = wf._fit_predict

        def recording(train, test, features, target):
            observed.append((train["session_date"].max(),
                             test["session_date"].min()))
            return original(train, test, features, target)

        wf._fit_predict = recording
        try:
            wf.walk_forward(frame=panel, initial=200, step=40)
        finally:
            wf._fit_predict = original

        self.assertTrue(observed)
        for latest_train, earliest_test in observed:
            self.assertLess(latest_train, earliest_test)

    def test_the_training_window_expands(self):
        panel = synthetic_panel()
        _, windows = wf.walk_forward(frame=panel, initial=200, step=40)
        sizes = list(windows["train"])
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])

    def test_test_windows_do_not_overlap(self):
        panel = synthetic_panel()
        _, windows = wf.walk_forward(frame=panel, initial=200, step=40)
        for earlier, later in zip(windows.itertuples(), windows.iloc[1:].itertuples()):
            self.assertLess(earlier.to, later._1)

    def test_every_scored_row_appears_once(self):
        panel = synthetic_panel()
        predictions, _ = wf.walk_forward(frame=panel, initial=200, step=40)
        keys = predictions[["ticker", "session_date"]]
        self.assertEqual(len(keys), len(keys.drop_duplicates()))

    def test_refuses_a_sample_too_short_to_split(self):
        with self.assertRaises(SystemExit):
            wf.walk_forward(frame=synthetic_panel(sessions=50), initial=200, step=40)


class InformationCoefficient(unittest.TestCase):
    def frame_with(self, correlation_sign):
        """Predictions whose score orders names the same way as their return."""
        generator = np.random.default_rng(1)
        rows = []
        for date in pd.bdate_range("2015-01-01", periods=60):
            for name in range(10):
                excess = generator.normal(0, 0.02)
                rows.append({
                    "ticker": f"T{name:02d}",
                    "session_date": date,
                    "excess_return": excess,
                    "score": correlation_sign * excess,
                })
        return pd.DataFrame(rows)

    def test_a_perfect_ranking_scores_one(self):
        series = wf.information_coefficient(self.frame_with(1))
        self.assertAlmostEqual(series.mean(), 1.0, places=6)

    def test_an_inverted_ranking_scores_minus_one(self):
        series = wf.information_coefficient(self.frame_with(-1))
        self.assertAlmostEqual(series.mean(), -1.0, places=6)

    def test_stride_thins_the_series_for_overlapping_horizons(self):
        frame = self.frame_with(1)
        every = wf.information_coefficient(frame, stride=1)
        strided = wf.information_coefficient(frame, stride=5)
        # Consecutive five-day returns share four days, so their ICs are
        # autocorrelated; keeping every fifth session restores independence at the
        # cost of sample size, and an uncorrected count would overstate t by
        # roughly the square root of the stride.
        self.assertEqual(len(strided), len(range(0, len(every), 5)))
        self.assertLess(len(strided), len(every))

    def test_thin_sessions_are_excluded(self):
        frame = self.frame_with(1)
        # Leave two names on one session, below the minimum for a meaningful rank
        # correlation.
        first = frame["session_date"].min()
        thin = frame[(frame["session_date"] != first)
                     | (frame["ticker"].isin(["T00", "T01"]))]
        series = wf.information_coefficient(thin)
        self.assertNotIn(first, series.index)


class Summary(unittest.TestCase):
    def test_baseline_is_the_majority_class(self):
        frame = pd.DataFrame({
            "ticker": ["A"] * 10,
            "session_date": pd.bdate_range("2015-01-01", periods=10),
            "score": [0.9] * 10,
            "excess_return": [0.01] * 10,
            "target_relative": [1] * 7 + [0] * 3,
        })
        stats = wf.summarize(frame)
        self.assertAlmostEqual(stats["baseline"], 0.7)
        # Every score is above 0.5, so the model predicts the majority every time.
        self.assertAlmostEqual(stats["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
