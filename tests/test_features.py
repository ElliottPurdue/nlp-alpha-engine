"""Feature construction.

The surprise baselines are the highest-stakes code in the project. They are built
from a rolling window shifted one observation back, and dropping that shift would
measure every row against a baseline containing itself -- a lookahead so small it
would never look wrong and would inflate every result downstream.

Also covered: the forward merge that resolves market holidays, and the
cross-sectional construction that the model's features depend on.
"""

import datetime as dt
import pathlib
import tempfile
import unittest

import build_features as bf
import database as db


class FeatureFixture(unittest.TestCase):
    """A database with a controlled panel: known sentiment against known prices."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = pathlib.Path(self._directory.name) / "test.db"
        self._article_counter = 0
        with db.connect(self.path) as conn:
            db.init_db(conn)

    def tearDown(self):
        self._directory.cleanup()

    def add_news(self, conn, ticker, session_date, label, confidence):
        """One article for one ticker, pinned to the given session.

        Timestamped at 13:00 UTC (09:00 ET) so to_session_date leaves the date
        alone -- the roll is exercised elsewhere, not here.

        The link carries a counter so that repeated calls create distinct
        articles. Deriving it from ticker and date alone made several calls share
        a URL, and the dedup then collapsed them into one article -- correct
        behaviour that quietly defeated any test counting headlines.
        """
        self._article_counter += 1
        link = (f"https://example.com/{ticker}/{session_date}/"
                f"{label}/{self._article_counter}")
        db.upsert_articles(conn, [{
            "ticker": ticker,
            "headline": f"{ticker} {session_date}",
            "published_at": dt.datetime.fromisoformat(f"{session_date}T13:00:00+00:00"),
            "link": link,
        }])
        article_id = conn.execute(
            "SELECT article_id FROM raw_news WHERE url = ?", (link,)).fetchone()[0]
        db.upsert_sentiment(conn, [(article_id, label, confidence)])

    def add_prices(self, conn, ticker, closes):
        """closes maps YYYY-MM-DD to a closing price."""
        db.upsert_prices(conn, [
            (ticker, day, close, close, close, close, 1_000_000)
            for day, close in closes.items()
        ])

    def features(self, conn):
        bf.rebuild_features(conn)
        return {(r["ticker"], r["session_date"]): r for r in conn.execute(
            "SELECT * FROM daily_features")}


class SurpriseBaselineIsCausal(FeatureFixture):
    """The decisive test: a baseline must never include the row it measures."""

    # 2020-07-01 through 07-10 are Wed..Fri across two weeks, all trading days.
    SESSIONS = ["2020-07-01", "2020-07-02", "2020-07-06", "2020-07-07",
                "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-13"]

    def build(self, conn):
        # Five quiet sessions at neutral, then one strongly positive.
        for index, session in enumerate(self.SESSIONS):
            if index == 5:
                self.add_news(conn, "AAPL", session, "positive", 1.0)
            else:
                self.add_news(conn, "AAPL", session, "neutral", 0.5)
        self.add_prices(conn, "AAPL",
                        {s: 100.0 + i for i, s in enumerate(self.SESSIONS)})

    def test_the_spike_is_measured_against_the_quiet_days_alone(self):
        with db.connect(self.path) as conn:
            self.build(conn)
            rows = self.features(conn)

        spike = rows[("AAPL", self.SESSIONS[5])]

        # Five prior sessions all scored neutral, so net_sentiment 0.0 and the
        # baseline is 0.0. The spike is 1.0, so the surprise is the full 1.0.
        #
        # Without the shift, the rolling window would include the spike itself:
        # the baseline would be 1/6 and the surprise would read about 0.833. That
        # is the number this assertion exists to exclude.
        self.assertAlmostEqual(spike["mean_sentiment"], 1.0, places=6)
        self.assertAlmostEqual(spike["sentiment_surprise"], 1.0, places=6)
        self.assertNotAlmostEqual(spike["sentiment_surprise"], 5.0 / 6.0, places=3)

    def test_early_sessions_have_no_baseline_at_all(self):
        with db.connect(self.path) as conn:
            self.build(conn)
            rows = self.features(conn)

        # MIN_BASELINE prior observations are required, so the first four sessions
        # cannot have one. Leaving them NULL is correct; inventing a baseline from
        # one or two points would be worse than having none.
        for session in self.SESSIONS[:bf.MIN_BASELINE - 1]:
            self.assertIsNone(rows[("AAPL", session)]["sentiment_surprise"])
        self.assertIsNotNone(rows[("AAPL", self.SESSIONS[5])]["sentiment_surprise"])

    def test_attention_surprise_is_also_measured_against_prior_days(self):
        with db.connect(self.path) as conn:
            # One article a day, then a burst of four on the sixth session.
            for index, session in enumerate(self.SESSIONS):
                copies = 4 if index == 5 else 1
                for copy in range(copies):
                    self.add_news(conn, "AAPL", session, "neutral", 0.5 + copy / 100)
            self.add_prices(conn, "AAPL",
                            {s: 100.0 + i for i, s in enumerate(self.SESSIONS)})
            rows = self.features(conn)

        burst = rows[("AAPL", self.SESSIONS[5])]
        self.assertEqual(burst["headline_count"], 4)
        # log((4+1)/(1+1)) with a trailing median of 1, positive and finite.
        self.assertGreater(burst["attention_surprise"], 0.0)


class HolidayRoll(FeatureFixture):
    """session_date is derived without an exchange calendar, so it can land on a
    day the market was shut. The feature build advances it to the next date that
    actually traded."""

    def test_news_dated_on_a_closed_day_moves_to_the_next_open_one(self):
        with db.connect(self.path) as conn:
            # 2020-07-03 was a Friday and the observed Independence Day holiday.
            self.add_news(conn, "AAPL", "2020-07-02", "positive", 0.8)
            self.add_news(conn, "AAPL", "2020-07-03", "negative", 0.9)
            # No price bar for the 3rd, as on a real holiday.
            self.add_prices(conn, "AAPL", {"2020-07-02": 100.0,
                                           "2020-07-06": 102.0,
                                           "2020-07-07": 103.0})
            rows = self.features(conn)

        self.assertNotIn(("AAPL", "2020-07-03"), rows)
        self.assertIn(("AAPL", "2020-07-06"), rows)
        self.assertAlmostEqual(rows[("AAPL", "2020-07-06")]["mean_sentiment"],
                               -0.9, places=6)

    def test_two_dates_landing_on_one_session_are_recombined_not_averaged(self):
        with db.connect(self.path) as conn:
            # Holiday news plus the next session's own news, both on the 6th.
            self.add_news(conn, "AAPL", "2020-07-03", "positive", 1.0)
            self.add_news(conn, "AAPL", "2020-07-06", "neutral", 0.5)
            self.add_news(conn, "AAPL", "2020-07-06", "neutral", 0.6)
            self.add_prices(conn, "AAPL", {"2020-07-06": 100.0, "2020-07-07": 101.0})
            rows = self.features(conn)

        combined = rows[("AAPL", "2020-07-06")]
        # Three articles pooled: totals summed, then the mean taken once. Averaging
        # the two groups' means instead would give 0.5 rather than 1/3.
        self.assertEqual(combined["headline_count"], 3)
        self.assertAlmostEqual(combined["sum_sentiment"], 1.0, places=6)
        self.assertAlmostEqual(combined["mean_sentiment"], 1.0 / 3.0, places=6)


class CrossSectionalConstruction(FeatureFixture):
    def build_two_names(self, conn):
        sessions = ["2020-07-01", "2020-07-02", "2020-07-06"]
        for session in sessions:
            self.add_news(conn, "AAA", session, "positive", 0.9)
            self.add_news(conn, "BBB", session, "negative", 0.9)
        # AAA rises, BBB falls, so their excess returns are equal and opposite.
        self.add_prices(conn, "AAA", dict(zip(sessions, [100.0, 110.0, 121.0])))
        self.add_prices(conn, "BBB", dict(zip(sessions, [100.0, 90.0, 81.0])))

    def test_excess_return_is_zero_mean_within_each_session(self):
        with db.connect(self.path) as conn:
            self.build_two_names(conn)
            bf.rebuild_features(conn)
            sums = conn.execute(
                "SELECT session_date, SUM(excess_return) AS total"
                " FROM daily_features WHERE excess_return IS NOT NULL"
                " GROUP BY session_date").fetchall()

        self.assertTrue(sums)
        for row in sums:
            self.assertAlmostEqual(row["total"], 0.0, places=9)

    def test_ranks_are_percentiles(self):
        with db.connect(self.path) as conn:
            self.build_two_names(conn)
            bf.rebuild_features(conn)
            rows = conn.execute(
                "SELECT sentiment_rank, headline_rank, volume_rank"
                " FROM daily_features").fetchall()

        for row in rows:
            for column in ("sentiment_rank", "headline_rank", "volume_rank"):
                self.assertGreater(row[column], 0.0)
                self.assertLessEqual(row[column], 1.0)

    def test_the_latest_session_has_no_forward_label(self):
        with db.connect(self.path) as conn:
            self.build_two_names(conn)
            bf.rebuild_features(conn)
            unlabelled = conn.execute(
                "SELECT COUNT(*) FROM daily_features"
                " WHERE target_relative IS NULL").fetchone()[0]
            tickers = conn.execute(
                "SELECT COUNT(DISTINCT ticker) FROM daily_features").fetchone()[0]

        # Exactly one row per ticker lacks a forward return: its most recent
        # session. This invariant has held on every real rebuild too.
        self.assertEqual(unlabelled, tickers)


if __name__ == "__main__":
    unittest.main()
