"""Upsert behaviour and the guarantees the schema is supposed to enforce.

Idempotent writes are what let the scraper run hourly against a feed that repeats
itself and the scorer resume after an interruption. If an upsert silently became an
insert, history would inflate without any error appearing.
"""

import pathlib
import sqlite3
import tempfile
import unittest

import database as db


class TemporaryDatabase(unittest.TestCase):
    """Each test gets its own database file, so none can see another's writes."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = pathlib.Path(self._directory.name) / "test.db"
        with db.connect(self.path) as conn:
            db.init_db(conn)

    def tearDown(self):
        self._directory.cleanup()

    def article(self, link="https://example.com/a", ticker="AAPL",
                headline="Headline", published="Wed, 15 Jul 2020 13:00:00 +0000"):
        return {"ticker": ticker, "headline": headline,
                "published_at": published, "link": link}


class Idempotency(TemporaryDatabase):
    def test_the_same_article_is_stored_once(self):
        with db.connect(self.path) as conn:
            first = db.upsert_articles(conn, [self.article()])
            second = db.upsert_articles(conn, [self.article()])
            total = db.table_counts(conn)["raw_news"]

        self.assertEqual(first["articles_new"], 1)
        self.assertEqual(second["articles_new"], 0)
        self.assertEqual(total, 1)

    def test_equivalent_urls_are_the_same_article(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article("https://www.example.com/a?utm=1")])
            db.upsert_articles(conn, [self.article("https://example.com/a/")])
            self.assertEqual(db.table_counts(conn)["raw_news"], 1)

    def test_first_seen_survives_but_last_seen_advances(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article()])
            before = conn.execute(
                "SELECT first_seen_at, last_seen_at FROM raw_news").fetchone()
            # Force a distinguishable timestamp on the second write.
            conn.execute("UPDATE raw_news SET last_seen_at = '1999-01-01T00:00:00'")
            db.upsert_articles(conn, [self.article()])
            after = conn.execute(
                "SELECT first_seen_at, last_seen_at FROM raw_news").fetchone()

        self.assertEqual(before["first_seen_at"], after["first_seen_at"])
        self.assertNotEqual(after["last_seen_at"], "1999-01-01T00:00:00")

    def test_headline_text_is_not_rewritten_by_a_rescrape(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article(headline="Original")])
            db.upsert_articles(conn, [self.article(headline="Edited later")])
            stored = conn.execute("SELECT headline FROM raw_news").fetchone()[0]
        self.assertEqual(stored, "Original")


class Syndication(TemporaryDatabase):
    def test_one_article_across_two_tickers_is_stored_once(self):
        records = [self.article(ticker="AAPL"), self.article(ticker="MSFT")]
        with db.connect(self.path) as conn:
            stats = db.upsert_articles(conn, records)
            counts = db.table_counts(conn)

        self.assertEqual(counts["raw_news"], 1)
        self.assertEqual(counts["news_tickers"], 2)
        self.assertEqual(stats["articles_new"], 1)
        self.assertEqual(stats["links_new"], 2)

    def test_repeated_ticker_links_do_not_duplicate(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article(ticker="AAPL")])
            db.upsert_articles(conn, [self.article(ticker="AAPL")])
            self.assertEqual(db.table_counts(conn)["news_tickers"], 1)


class GeneratedSentiment(TemporaryDatabase):
    """net_sentiment is defined in the schema so every consumer agrees on it."""

    def score(self, label, confidence):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article()])
            article_id = conn.execute("SELECT article_id FROM raw_news").fetchone()[0]
            db.upsert_sentiment(conn, [(article_id, label, confidence)])
            return conn.execute(
                "SELECT net_sentiment FROM sentiment_scores").fetchone()[0]

    def test_positive_keeps_the_confidence(self):
        self.assertAlmostEqual(self.score("positive", 0.87), 0.87)

    def test_negative_inverts_it(self):
        self.assertAlmostEqual(self.score("negative", 0.91), -0.91)

    def test_neutral_is_zero_regardless_of_confidence(self):
        self.assertAlmostEqual(self.score("neutral", 0.55), 0.0)

    def test_rescoring_replaces_rather_than_duplicates(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article()])
            article_id = conn.execute("SELECT article_id FROM raw_news").fetchone()[0]
            db.upsert_sentiment(conn, [(article_id, "positive", 0.6)])
            db.upsert_sentiment(conn, [(article_id, "negative", 0.8)])
            rows = conn.execute(
                "SELECT sentiment_label, net_sentiment FROM sentiment_scores").fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sentiment_label"], "negative")
        self.assertAlmostEqual(rows[0]["net_sentiment"], -0.8)


class SchemaGuarantees(TemporaryDatabase):
    def insert_score(self, label="positive", confidence=0.5, article_id=None):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article()])
            if article_id is None:
                article_id = conn.execute(
                    "SELECT article_id FROM raw_news").fetchone()[0]
            conn.execute(
                "INSERT INTO sentiment_scores (article_id, model_name,"
                " sentiment_label, sentiment_score, scored_at)"
                " VALUES (?, 'm', ?, ?, 'now')",
                (article_id, label, confidence),
            )

    def test_rejects_an_unknown_sentiment_label(self):
        # A model version returning 'bullish' would otherwise score every headline
        # as neutral, which reads as a finding rather than a bug.
        with self.assertRaises(sqlite3.IntegrityError):
            self.insert_score(label="bullish")

    def test_rejects_a_confidence_outside_zero_to_one(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.insert_score(confidence=1.4)

    def test_rejects_a_score_for_an_article_that_does_not_exist(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.insert_score(article_id=9999)

    def test_rejects_text_in_a_numeric_column(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with db.connect(self.path) as conn:
                conn.execute("INSERT INTO prices VALUES ('AAPL','2020-07-15',"
                             "1.0,1.0,1.0,'n/a',100)")

    def test_refuses_a_written_value_for_the_generated_column(self):
        with self.assertRaises(sqlite3.OperationalError):
            with db.connect(self.path) as conn:
                db.upsert_articles(conn, [self.article()])
                conn.execute(
                    "INSERT INTO sentiment_scores (article_id, model_name,"
                    " sentiment_label, sentiment_score, net_sentiment, scored_at)"
                    " VALUES (1,'m','positive',0.5,99.0,'now')")

    def test_deleting_an_article_removes_its_dependents(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article(ticker="AAPL"),
                                      self.article(ticker="MSFT")])
            article_id = conn.execute("SELECT article_id FROM raw_news").fetchone()[0]
            db.upsert_sentiment(conn, [(article_id, "positive", 0.5)])
            conn.execute("DELETE FROM raw_news")
            counts = db.table_counts(conn)

        self.assertEqual(counts["news_tickers"], 0)
        self.assertEqual(counts["sentiment_scores"], 0)


class UnscoredBacklog(TemporaryDatabase):
    """The anti-join is what makes scoring resumable and cheap to re-run."""

    def test_returns_only_articles_without_a_score(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article("https://example.com/a"),
                                      self.article("https://example.com/b")])
            self.assertEqual(len(db.fetch_unscored_articles(conn)), 2)

            first = db.fetch_unscored_articles(conn)[0]["article_id"]
            db.upsert_sentiment(conn, [(first, "positive", 0.5)])
            remaining = db.fetch_unscored_articles(conn)

        self.assertEqual(len(remaining), 1)
        self.assertNotEqual(remaining[0]["article_id"], first)

    def test_a_score_from_another_model_does_not_count(self):
        with db.connect(self.path) as conn:
            db.upsert_articles(conn, [self.article()])
            article_id = conn.execute("SELECT article_id FROM raw_news").fetchone()[0]
            db.upsert_sentiment(conn, [(article_id, "positive", 0.5)],
                                model_name="some-other-model")
            self.assertEqual(len(db.fetch_unscored_articles(conn)), 1)


if __name__ == "__main__":
    unittest.main()
