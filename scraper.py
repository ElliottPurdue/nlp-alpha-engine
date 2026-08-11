"""Stage 1: collect Yahoo Finance RSS headlines into SQLite.

Writes through an upsert, so the scraper can run at any frequency without
duplicating rows. Each feed only exposes its most recent items, so how often this
runs determines how much of the news flow is captured.
"""

import time

import requests
from bs4 import BeautifulSoup

import database as db

# Spread across sectors rather than concentrated in mega-cap tech. This is a
# cross-sectional signal, and names that move together add far less information
# than their row count suggests.
UNIVERSE = {
    "technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "AMD",
        "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "TXN", "INTC",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    ],
    "healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    ],
    "consumer": [
        "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "NKE", "TGT",
    ],
    "industrials": [
        "CAT", "BA", "GE", "UPS", "HON", "RTX",
    ],
    "energy": [
        "XOM", "CVX", "COP",
    ],
    "communications": [
        "DIS", "NFLX", "T", "VZ",
    ],
    "autos": [
        "TSLA", "F", "GM",
    ],
}

TICKERS = [ticker for sector in UNIVERSE.values() for ticker in sector]

REQUEST_TIMEOUT = 15

# At 57 tickers, firing every request back to back is impolite and invites
# throttling.
REQUEST_DELAY = 0.5

# A full pass takes about ninety seconds. Without periodic output it looks hung.
PROGRESS_EVERY = 10

# The endpoint returns 403 without a browser User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def fetch_yahoo_news(ticker):
    """Fetch and parse one ticker's feed into ticker/headline/published_at/link dicts."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

    # Without this an error page reaches the parser, which finds no items and
    # reports an empty feed rather than a failed request.
    response.raise_for_status()

    soup = BeautifulSoup(response.content, features="xml")

    news_data = []
    for article in soup.find_all("item"):
        # No link means no dedup key; no date means no session. Drop rather than
        # store placeholders.
        if not article.link or not article.title or not article.pubDate:
            continue
        news_data.append({
            "ticker": ticker,
            "headline": article.title.text,
            "published_at": article.pubDate.text,
            "link": article.link.text,
        })

    return news_data


def run():
    """Scrape every ticker and persist. Exits non-zero if nothing was collected."""
    all_news = []
    failed = []

    # Only failures are logged per ticker. A line per feed would add over a
    # thousand lines a day to pipeline.log and bury the summary.
    print(f"Fetching news for {len(TICKERS)} tickers...")
    for index, ticker in enumerate(TICKERS):
        if index:
            time.sleep(REQUEST_DELAY)

        try:
            all_news.extend(fetch_yahoo_news(ticker))
        except requests.RequestException as exc:
            print(f"  {ticker}: FAILED ({exc})")
            failed.append(ticker)

        completed = index + 1
        if completed % PROGRESS_EVERY == 0 or completed == len(TICKERS):
            print(f"  {completed}/{len(TICKERS)} feeds | {len(all_news)} items collected")

    with db.connect() as conn:
        db.init_db(conn)
        stats = db.upsert_articles(conn, all_news)
        total = db.table_counts(conn)["raw_news"]

    print(f"\nScraped {len(all_news)} feed items -> {stats['articles_seen']} unique articles")
    print(f"  {stats['articles_new']} new articles, {stats['links_new']} new ticker links")
    print(f"  {total} articles in database")
    if failed:
        print(f"  {len(failed)} of {len(TICKERS)} feeds unavailable: {', '.join(failed)}")

    # A scheduler only sees the exit code. Collecting nothing means blocked
    # requests, no connectivity, or a changed feed format, and must not be
    # reported as success. The two cases are separated because they need
    # different diagnoses.
    if not all_news:
        if len(failed) == len(TICKERS):
            raise SystemExit(f"FAILED: all {len(TICKERS)} feeds unreachable")
        raise SystemExit("FAILED: feeds responded but no items were parsed")


if __name__ == "__main__":
    run()
