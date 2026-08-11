"""SQLite persistence layer.

Every write is an idempotent upsert, so any stage can be re-run without
duplicating rows. Tables are defined in schema.sql.
"""

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

# Relative to this file, not the working directory, so scheduled tasks and the
# dashboard find the database wherever they are launched from.
PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "alpha_engine.db"
SCHEMA_PATH = PROJECT_DIR / "schema.sql"

FINBERT_MODEL = "ProsusAI/finbert"

EXCHANGE_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)

# SQLite caps bound parameters per statement; chunk lookups below the limit.
_PARAM_CHUNK = 900


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------

@contextmanager
def connect(db_path=DB_PATH, read_only=False):
    """Commit on clean exit, roll back on exception, always close."""
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)

    conn.row_factory = sqlite3.Row

    # Off by default, and per connection, so it cannot live in the schema.
    conn.execute("PRAGMA foreign_keys = ON")

    if not read_only:
        # WAL lets the dashboard read while a scheduled run is writing.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

    try:
        yield conn
        if not read_only:
            conn.commit()
    except Exception:
        if not read_only:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn):
    """Apply schema.sql. All DDL is IF NOT EXISTS, so this is safe every run."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def canonical_url(url):
    """Strip query, fragment, leading www and trailing slash.

    Feed providers append tracking parameters that vary between requests for the
    same article, so the raw URL is not a usable identity.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), "", ""))


def url_fingerprint(url):
    """SHA-256 of the canonical URL. This is the article key."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()


def url_source(url):
    """Publisher host, e.g. finance.yahoo.com."""
    host = urlsplit(url.strip()).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def parse_published(raw):
    """Parse an RSS pubDate into aware UTC. Accepts a datetime unchanged.

    RSS mandates RFC 2822, which email.utils parses to spec. Bad input raises
    rather than falling back to a guess: a wrong timestamp silently corrupts the
    session assignment and everything derived from it.
    """
    dt = raw if isinstance(raw, datetime) else parsedate_to_datetime(str(raw).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_session_date(published_utc):
    """Trading session a headline is actionable in.

    A close-to-close position is entered at the 16:00 ET close, so only news
    published before it can inform that trade. Anything later, or outside the
    trading week, belongs to the next session. Applied to live feed data this
    reclassifies about half of all articles.

    Holidays are left alone here; build_features.py rolls each session_date onto
    the next date that actually appears in `prices`.
    """
    local = published_utc.astimezone(EXCHANGE_TZ)
    session = local.date()
    if local.time() >= MARKET_CLOSE:
        session += timedelta(days=1)
    while session.weekday() >= 5:
        session += timedelta(days=1)
    return session


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def upsert_articles(conn, records):
    """Upsert headlines and their ticker tags.

    `records` are dicts of ticker/headline/published_at/link, the shape the
    scraper emits. Articles carried by several feeds collapse to one raw_news row.

    Returns counts: articles_seen, articles_new, links_new.
    """
    now = _utc_now()
    articles = {}
    tags = set()

    for rec in records:
        link = rec["link"]
        if not link or link == "No Link":
            continue
        fp = url_fingerprint(link)
        published = parse_published(rec["published_at"])
        articles.setdefault(fp, (
            fp,
            link,
            rec["headline"],
            url_source(link),
            published.isoformat().replace("+00:00", "Z"),
            to_session_date(published).isoformat(),
            now,
            now,
        ))
        tags.add((fp, rec["ticker"]))

    if not articles:
        return {"articles_seen": 0, "articles_new": 0, "links_new": 0}

    before_articles = _count(conn, "raw_news")
    before_links = _count(conn, "news_tickers")

    # first_seen_at stays out of the update clause: it records first discovery and
    # must survive every later re-scrape.
    conn.executemany(
        """
        INSERT INTO raw_news (url_hash, url, headline, source,
                              published_at, session_date,
                              first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url_hash) DO UPDATE SET last_seen_at = excluded.last_seen_at
        """,
        list(articles.values()),
    )

    ids = article_ids_by_fingerprint(conn, articles.keys())
    conn.executemany(
        "INSERT OR IGNORE INTO news_tickers (article_id, ticker) VALUES (?, ?)",
        [(ids[fp], ticker) for fp, ticker in tags if fp in ids],
    )

    return {
        "articles_seen": len(articles),
        "articles_new": _count(conn, "raw_news") - before_articles,
        "links_new": _count(conn, "news_tickers") - before_links,
    }


def upsert_sentiment(conn, scores, model_name=FINBERT_MODEL):
    """Upsert (article_id, label, confidence) triples for one model."""
    scores = list(scores)
    if not scores:
        return 0
    now = _utc_now()
    conn.executemany(
        """
        INSERT INTO sentiment_scores (article_id, model_name, sentiment_label,
                                      sentiment_score, scored_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(article_id, model_name) DO UPDATE SET
            sentiment_label = excluded.sentiment_label,
            sentiment_score = excluded.sentiment_score,
            scored_at       = excluded.scored_at
        """,
        [(aid, model_name, label.lower(), float(score), now)
         for aid, label, score in scores],
    )
    return len(scores)


def upsert_prices(conn, rows):
    """Upsert (ticker, date, open, high, low, close, volume) tuples."""
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open   = excluded.open,
            high   = excluded.high,
            low    = excluded.low,
            close  = excluded.close,
            volume = excluded.volume
        """,
        rows,
    )
    return len(rows)


def replace_daily_features(conn, rows):
    """Swap in a fresh feature matrix.

    Rebuilt rather than merged so rows computed under an old feature definition
    cannot survive next to current ones. The delete and insert share the caller's
    transaction, so a reader never sees the table empty.
    """
    rows = list(rows)
    conn.execute("DELETE FROM daily_features")
    if not rows:
        return 0
    now = _utc_now()
    conn.executemany(
        """
        INSERT INTO daily_features (ticker, session_date, mean_sentiment,
                                    sum_sentiment, headline_count, close, volume,
                                    sentiment_rank, headline_rank, volume_rank,
                                    sentiment_surprise, attention_surprise,
                                    sentiment_surprise_rank, attention_surprise_rank,
                                    fwd_return, excess_return, target,
                                    target_relative, fwd_return_5d,
                                    excess_return_5d, target_relative_5d, built_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(r) + (now,) for r in rows],
    )
    return len(rows)


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def article_ids_by_fingerprint(conn, fingerprints):
    """Map url_hash to article_id, chunked around the parameter cap."""
    fingerprints = list(fingerprints)
    out = {}
    for i in range(0, len(fingerprints), _PARAM_CHUNK):
        chunk = fingerprints[i:i + _PARAM_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT url_hash, article_id FROM raw_news WHERE url_hash IN ({placeholders})",
            chunk,
        ).fetchall()
        out.update({r["url_hash"]: r["article_id"] for r in rows})
    return out


def fetch_unscored_articles(conn, model_name=FINBERT_MODEL):
    """Articles with no score for this model.

    The anti-join is what makes scoring cheap to re-run and resumable: inference
    dominates runtime, and headline text is already stored, so a backlog of any
    size can be cleared whenever convenient.
    """
    return conn.execute(
        """
        SELECT n.article_id, n.headline
        FROM raw_news n
        LEFT JOIN sentiment_scores s
               ON s.article_id = n.article_id AND s.model_name = ?
        WHERE s.article_id IS NULL
        ORDER BY n.article_id
        """,
        (model_name,),
    ).fetchall()


def distinct_tickers(conn):
    rows = conn.execute("SELECT DISTINCT ticker FROM news_tickers ORDER BY ticker").fetchall()
    return [r["ticker"] for r in rows]


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def table_counts(conn):
    return {t: _count(conn, t) for t in
            ("raw_news", "news_tickers", "sentiment_scores", "prices", "daily_features")}


if __name__ == "__main__":
    with connect() as conn:
        init_db(conn)
        print(f"Initialized {DB_PATH}")
        for table, n in table_counts(conn).items():
            print(f"  {table:<18} {n:>6} rows")
