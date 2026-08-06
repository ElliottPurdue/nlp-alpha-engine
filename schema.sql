-- Schema for the NLP alpha engine.
--
-- The tables form two layers:
--
--   Source of truth (raw_news, news_tickers, sentiment_scores, prices)
--       Written only through idempotent upserts and never rebuilt. RSS feeds
--       expose a short rolling window of articles, so a record lost here cannot
--       be recovered from upstream.
--
--   Derived (daily_features)
--       Dropped and regenerated from the source tables on every feature build.
--
-- Conventions:
--
--   Dates and timestamps are stored as ISO-8601 TEXT. SQLite has no native date
--   type, and ISO-8601 orders lexicographically, so range predicates and ORDER BY
--   behave correctly on plain strings.
--
--   All tables are STRICT, which rejects values whose type does not match the
--   column declaration. STRICT does not catch NaN: SQLite has no NaN
--   representation and coerces it to NULL before type checking. NOT NULL is the
--   constraint that rejects a missing numeric value, which is why `close` is
--   NOT NULL while `volume`, where a gap is tolerable, is not.


-- Unique articles, deduplicated on a hash of the canonicalized URL.
--
-- Rows are immutable apart from last_seen_at: re-scraping an article still
-- present in the feed refreshes that column and nothing else.
CREATE TABLE IF NOT EXISTS raw_news (
    article_id     INTEGER PRIMARY KEY,
    url_hash       TEXT NOT NULL UNIQUE,   -- SHA-256 of scheme + host + path
    url            TEXT NOT NULL,          -- original link, unmodified
    headline       TEXT NOT NULL,
    source         TEXT NOT NULL,          -- publisher host
    published_at   TEXT NOT NULL,          -- ISO-8601 UTC, as reported by the feed
    session_date   TEXT NOT NULL,          -- trading session derived from published_at
    first_seen_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL
) STRICT;


-- Associates articles with the ticker feeds that carried them. An article
-- syndicated across several feeds has one row in raw_news and one row here per
-- ticker.
CREATE TABLE IF NOT EXISTS news_tickers (
    article_id INTEGER NOT NULL REFERENCES raw_news(article_id) ON DELETE CASCADE,
    ticker     TEXT    NOT NULL,
    PRIMARY KEY (article_id, ticker)
) STRICT, WITHOUT ROWID;


-- Classifier output, keyed by model so that scores from different models
-- coexist rather than overwrite one another.
--
-- net_sentiment is a generated column: the mapping from label and confidence to
-- a signed magnitude is defined once, here, so every consumer of the data
-- resolves it identically.
CREATE TABLE IF NOT EXISTS sentiment_scores (
    article_id      INTEGER NOT NULL REFERENCES raw_news(article_id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,
    sentiment_label TEXT NOT NULL CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
    sentiment_score REAL NOT NULL CHECK (sentiment_score BETWEEN 0.0 AND 1.0),
    net_sentiment   REAL GENERATED ALWAYS AS (
        CASE sentiment_label
            WHEN 'positive' THEN  sentiment_score
            WHEN 'negative' THEN -sentiment_score
            ELSE 0.0
        END
    ) STORED,
    scored_at       TEXT NOT NULL,
    PRIMARY KEY (article_id, model_name)
) STRICT, WITHOUT ROWID;


-- Daily OHLCV bars. Upserted rather than insert-ignored, because vendors restate
-- historical values following splits and dividends and the later value
-- supersedes the earlier one.
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,   -- YYYY-MM-DD, exchange-local trading date
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL NOT NULL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
) STRICT, WITHOUT ROWID;


-- Model-ready feature matrix, regenerated in full by the feature build.
--
-- fwd_return and target are NULL for each ticker's most recent session, which
-- has no subsequent close. Those rows are retained so that current sentiment
-- remains available for display, and are excluded at training time.
CREATE TABLE IF NOT EXISTS daily_features (
    ticker         TEXT NOT NULL,
    session_date   TEXT NOT NULL,
    mean_sentiment REAL NOT NULL,
    sum_sentiment  REAL NOT NULL,
    headline_count INTEGER NOT NULL,
    close          REAL NOT NULL,
    volume         INTEGER,
    fwd_return     REAL,       -- close(D+1) / close(D) - 1
    target         INTEGER,    -- 1 if fwd_return > 0, else 0
    built_at       TEXT NOT NULL,
    PRIMARY KEY (ticker, session_date)
) STRICT, WITHOUT ROWID;


-- session_date drives feature builds and dashboard date filters; published_at
-- drives recency ordering; ticker drives every per-symbol query.
CREATE INDEX IF NOT EXISTS idx_raw_news_session   ON raw_news(session_date);
CREATE INDEX IF NOT EXISTS idx_raw_news_published ON raw_news(published_at);
CREATE INDEX IF NOT EXISTS idx_news_tickers_tkr   ON news_tickers(ticker);


-- Denormalized headline feed for presentation layers.
--
-- LEFT JOIN on sentiment_scores so that articles collected but not yet scored
-- still appear, with NULL sentiment, rather than being filtered out between
-- pipeline stages.
CREATE VIEW IF NOT EXISTS v_scored_headlines AS
SELECT n.article_id,
       t.ticker,
       n.headline,
       n.url,
       n.source,
       n.published_at,
       n.session_date,
       s.model_name,
       s.sentiment_label,
       s.sentiment_score,
       s.net_sentiment
FROM raw_news n
JOIN news_tickers t USING (article_id)
LEFT JOIN sentiment_scores s USING (article_id);
