"""Trading-session attribution and timestamp parsing.

to_session_date decides which session a headline is allowed to inform. If it lets
post-close news into the session that just ended, the model trains on information
that did not exist when the trade would have been placed -- and no accuracy metric
reveals it. The boundaries are pinned here.
"""

import datetime as dt
import unittest

import database as db


def utc(text):
    """ISO-8601 string to aware datetime."""
    return dt.datetime.fromisoformat(text)


class CloseBoundary(unittest.TestCase):
    """2020-07-15 is a Wednesday; 20:00Z that day is exactly 16:00 EDT."""

    def test_before_close_stays_in_the_same_session(self):
        self.assertEqual(db.to_session_date(utc("2020-07-15T19:59:00+00:00")),
                         dt.date(2020, 7, 15))

    def test_at_the_close_rolls_forward(self):
        # Inclusive on purpose: news arriving exactly at the close cannot inform a
        # position entered at that close.
        self.assertEqual(db.to_session_date(utc("2020-07-15T20:00:00+00:00")),
                         dt.date(2020, 7, 16))

    def test_after_close_rolls_forward(self):
        self.assertEqual(db.to_session_date(utc("2020-07-15T23:30:00+00:00")),
                         dt.date(2020, 7, 16))

    def test_early_morning_belongs_to_that_day(self):
        # 09:00Z is 05:00 EDT, before the open but before the close, so it informs
        # the session about to begin.
        self.assertEqual(db.to_session_date(utc("2020-07-15T09:00:00+00:00")),
                         dt.date(2020, 7, 15))


class DaylightSaving(unittest.TestCase):
    """The close is 16:00 exchange time, which is a moving target in UTC.

    A hardcoded UTC cutoff would misclassify several months of every year. These
    two cases sit on opposite sides of the daylight-saving shift.
    """

    def test_summer_close_falls_at_2000_utc(self):
        self.assertEqual(db.to_session_date(utc("2020-07-15T19:59:00+00:00")),
                         dt.date(2020, 7, 15))
        self.assertEqual(db.to_session_date(utc("2020-07-15T20:00:00+00:00")),
                         dt.date(2020, 7, 16))

    def test_winter_close_falls_an_hour_later(self):
        # 2020-01-15 is a Wednesday. 20:00Z is 15:00 EST, still before the close.
        self.assertEqual(db.to_session_date(utc("2020-01-15T20:00:00+00:00")),
                         dt.date(2020, 1, 15))
        self.assertEqual(db.to_session_date(utc("2020-01-15T21:00:00+00:00")),
                         dt.date(2020, 1, 16))


class WeekendRoll(unittest.TestCase):
    """2020-07-17 is a Friday, 18th Saturday, 19th Sunday, 20th Monday."""

    def test_friday_before_close_stays_on_friday(self):
        self.assertEqual(db.to_session_date(utc("2020-07-17T15:00:00+00:00")),
                         dt.date(2020, 7, 17))

    def test_friday_after_close_becomes_monday(self):
        self.assertEqual(db.to_session_date(utc("2020-07-17T21:00:00+00:00")),
                         dt.date(2020, 7, 20))

    def test_saturday_becomes_monday(self):
        self.assertEqual(db.to_session_date(utc("2020-07-18T14:00:00+00:00")),
                         dt.date(2020, 7, 20))

    def test_sunday_becomes_monday(self):
        self.assertEqual(db.to_session_date(utc("2020-07-19T14:00:00+00:00")),
                         dt.date(2020, 7, 20))

    def test_sunday_after_close_still_becomes_monday(self):
        # The close roll and the weekend roll compose: Sunday 21:00Z advances to
        # Monday, which is already a weekday, so it stops there rather than
        # skipping to Tuesday.
        self.assertEqual(db.to_session_date(utc("2020-07-19T21:00:00+00:00")),
                         dt.date(2020, 7, 20))


class PublicationTimestamps(unittest.TestCase):
    def test_parses_rfc_2822_as_rss_requires(self):
        parsed = db.parse_published("Wed, 15 Jul 2020 20:33:16 +0000")
        self.assertEqual(parsed, utc("2020-07-15T20:33:16+00:00"))

    def test_converts_a_non_utc_offset(self):
        parsed = db.parse_published("Wed, 15 Jul 2020 16:33:16 -0400")
        self.assertEqual(parsed, utc("2020-07-15T20:33:16+00:00"))

    def test_passes_an_aware_datetime_through(self):
        moment = utc("2020-07-15T20:33:16+00:00")
        self.assertEqual(db.parse_published(moment), moment)

    def test_assumes_utc_for_a_naive_datetime(self):
        naive = dt.datetime(2020, 7, 15, 20, 33, 16)
        self.assertEqual(db.parse_published(naive), utc("2020-07-15T20:33:16+00:00"))

    def test_raises_rather_than_guessing_on_malformed_input(self):
        # A silently wrong timestamp corrupts the session assignment and every
        # feature derived from it, so failing loudly is the correct behaviour.
        with self.assertRaises(Exception):
            db.parse_published("No Date")


if __name__ == "__main__":
    unittest.main()
