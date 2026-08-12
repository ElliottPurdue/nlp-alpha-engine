"""Article identity.

The fingerprint is the dedup key for the whole pipeline. Too strict and the same
article is stored and scored repeatedly under trivially different URLs; too loose
and distinct articles collapse into one.
"""

import unittest

import database as db


class CanonicalUrl(unittest.TestCase):
    def test_strips_tracking_query(self):
        # Yahoo appends ?.tsrc=rss to every link it serves.
        self.assertEqual(
            db.canonical_url("https://finance.yahoo.com/news/story.html?.tsrc=rss"),
            "https://finance.yahoo.com/news/story.html",
        )

    def test_strips_leading_www(self):
        self.assertEqual(db.canonical_url("https://www.trefis.com/a/1"),
                         db.canonical_url("https://trefis.com/a/1"))

    def test_ignores_a_trailing_slash(self):
        self.assertEqual(db.canonical_url("https://example.com/a/1/"),
                         db.canonical_url("https://example.com/a/1"))

    def test_lowercases_scheme_and_host_but_not_path(self):
        # Hosts are case-insensitive; paths are not, and folding them would merge
        # genuinely different articles.
        self.assertEqual(db.canonical_url("HTTPS://Example.COM/Story"),
                         "https://example.com/Story")

    def test_drops_the_fragment(self):
        self.assertEqual(db.canonical_url("https://example.com/a#section-2"),
                         "https://example.com/a")


class Fingerprint(unittest.TestCase):
    def test_equivalent_urls_share_a_fingerprint(self):
        variants = [
            "https://www.example.com/news/story?utm=1",
            "https://example.com/news/story/",
            "HTTPS://Example.com/news/story",
        ]
        fingerprints = {db.url_fingerprint(u) for u in variants}
        self.assertEqual(len(fingerprints), 1)

    def test_different_articles_do_not_collide(self):
        self.assertNotEqual(db.url_fingerprint("https://example.com/a"),
                            db.url_fingerprint("https://example.com/b"))

    def test_is_a_sha256_hex_digest(self):
        fingerprint = db.url_fingerprint("https://example.com/a")
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fingerprint))


class Publisher(unittest.TestCase):
    def test_extracts_the_host(self):
        self.assertEqual(db.url_source("https://finance.yahoo.com/news/x"),
                         "finance.yahoo.com")

    def test_normalizes_www_and_case(self):
        self.assertEqual(db.url_source("https://WWW.Benzinga.com/news/x"),
                         "benzinga.com")


if __name__ == "__main__":
    unittest.main()
