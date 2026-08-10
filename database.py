"""SQLite persistence layer for the NLP alpha engine.

All writes are idempotent upserts, so any pipeline stage may be re-run -- after a
failure, by a scheduler, or manually -- without duplicating existing rows. Table
definitions live in schema.sql.
"""

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

# Resolved relative to this module rather than the working directory, so that
# scheduled tasks and the dashboard locate the database regardless of where they
# are launched from.
PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "alpha_engine.db"
SCHEMA_PATH = PROJECT_DIR / "schema.sql"

FINBERT_MODEL = "ProsusAI/finbert"

EXCHANGE_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)

# SQLite caps the number of bound parameters in a single statement. Lookups are
# chunked well below the limit.
_PARAM_CHUNK = 900


# --------------------------------------------------------------------------
# Connection handling
# --------------------------------------------------------------------------

@contextmanager
def connect(db_path=DB_PATH, read_only=False):
    """Yield a configured connection, committing on success and rolling back on error.

    Args:
        db_path: Path to the SQLite database file.
        read_only: Open the database read-only, suppressing commit, rollback and
            the write-oriented pragmas.

    Yields:
        A sqlite3.Connection whose rows support access by column name.
    """
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)

    conn.row_factory = sqlite3.Row

    # Foreign key enforcement defaults to off and is configured per connection,
    # so it must be set here rather than in the schema.
    conn.execute("PRAGMA foreign_keys = ON")

    if not read_only:
        # Write-ahead logging admits concurrent readers alongside a single
        # writer, which the dashboard depends on while a scheduled run is
        # in progress.
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
    """Apply schema.sql. Safe to call on every run; all DDL is IF NOT EXISTS."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------

def canonical_url(url):
    """Reduce a URL to a form suitable for deduplication.

    Retains scheme, host and path. Discards the query string, fragment, a leading
    'www.' and trailing slashes, since feed providers append tracking parameters
    that vary between requests for the same article.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), "", ""))


def url_fingerprint(url):
    """Return the SHA-256 of a canonicalized URL, used as the article key."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()


def url_source(url):
    """Return the publisher host of a URL, for example 'finance.yahoo.com'."""
    host = urlsplit(url.strip()).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def parse_published(raw):
    """Parse an RSS pubDate into a timezone-aware UTC datetime.

    RSS mandates RFC 2822 date formatting, which email.utils parses to
    specification. Malformed input raises rather than falling back to a heuristic
    parser, because an incorrect timestamp propagates silently into the trading
    session assignment and every feature derived from it.
    """
    dt = raw if isinstance(raw, datetime) else parsedate_to_datetime(str(raw).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_session_date(published_utc):
    """Return the trading session in which a headline is actionable.

    A close-to-close position is established at the 16:00 ET close of session D,
    so only headlines published before that close can inform it. Headlines
    published after the close, or outside the trading week, are attributed to the
    following session. Without this adjustment a substantial share of feed output
    would be attributed to a session whose return had already been determined.

    Market holidays are not resolved here. The feature build advances each
    session_date to the next date present in `prices`, which covers holidays
    without requiring an exchange calendar.
    """
    local = published_utc.astimezone(EXCHANGE_TZ)
    session = local.date()
    if local.time() >= MARKET_CLOSE:
        session += timedelta(days=1)
    while session.weekday() >= 5:  # Saturday and Sunday
        session += timedelta(days=1)
    return session


def _utc_now():
    """Return the current UTC time as an ISO-8601 string at second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def upsert_articles(conn, records):
    """Insert scraped headlines and their ticker associations.

    Args:
        conn: An open connection.
        records: Iterable of mappings with keys ticker, headline, published_at
            and link, matching the shape the scraper emits. Articles carried by
            more than one ticker feed collapse into a single raw_news row.

    Returns:
        A dict with keys articles_seen, articles_new and links_new.
    """
    now = _utc_now()
    articles = {}   # fingerprint -> raw_news row
    tags = set()    # (fingerprint, ticker)

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

    # first_seen_at is excluded from the update clause: it records first
    # discovery by this pipeline and must survive subsequent re-scrapes.
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
    """Insert or replace classifier output for one model.

    Args:
        conn: An open connection.
        scores: Iterable of (article_id, label, confidence) triples.
        model_name: Identifier recorded alongside each score.

    Returns:
        The number of rows written.
    """
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
    """Insert or update OHLCV bars.

    Args:
        conn: An open connection.
        rows: Iterable of (ticker, date, open, high, low, close, volume) tuples.

    Returns:
        The number of rows written.
    """
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
    """Replace the feature matrix within the caller's transaction.

    The derived table is rebuilt rather than merged, so that rows computed under
    a superseded feature definition cannot persist alongside current ones. The
    delete and the insert share one transaction, so a concurrent reader never
    observes the table empty.

    Args:
        conn: An open connection.
        rows: Iterable of tuples matching the daily_features column order,
            excluding built_at, which is applied here.

    Returns:
        The number of rows written.
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
                                    fwd_return, excess_return, target,
                                    target_relative, built_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(r) + (now,) for r in rows],
    )
    return len(rows)


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def article_ids_by_fingerprint(conn, fingerprints):
    """Return a mapping of url_hash to article_id for the given fingerprints."""
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
    """Return articles that carry no score for the given model.

    The anti-join keeps the cost of the sentiment stage proportional to new input
    rather than to total history, which matters because model inference dominates
    pipeline runtime. It also allows scoring to be backfilled at any point after
    collection, since the headline text is already persisted.
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
    """Return every ticker that has at least one associated article."""
    rows = conn.execute("SELECT DISTINCT ticker FROM news_tickers ORDER BY ticker").fetchall()
    return [r["ticker"] for r in rows]


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def table_counts(conn):
    """Return row counts for every table, for pipeline logging."""
    return {t: _count(conn, t) for t in
            ("raw_news", "news_tickers", "sentiment_scores", "prices", "daily_features")}


if __name__ == "__main__":
    with connect() as conn:
        init_db(conn)
        print(f"Initialized {DB_PATH}")
        for table, n in table_counts(conn).items():
            print(f"  {table:<18} {n:>6} rows")
