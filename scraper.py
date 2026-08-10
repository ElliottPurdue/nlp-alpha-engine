"""Pipeline stage 1: collect Yahoo Finance RSS headlines into SQLite.

Articles are written to raw_news and news_tickers through an upsert, so the
scraper may run at any frequency without duplicating rows. Each feed exposes only
its most recent items, so execution frequency determines how much of the news
flow is captured; anything that rotates out before a run is unrecoverable.
"""

import time

import requests
from bs4 import BeautifulSoup

import database as db

# The universe deliberately spans sectors rather than concentrating in mega-cap
# technology. This is a cross-sectional signal, and names that all move together
# contribute far less independent information than their row count suggests, so
# breadth across sectors is what makes additional tickers worth collecting.
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

# Brief pause between feed requests. At this universe size, issuing every request
# back to back is both impolite to the provider and more likely to be throttled.
REQUEST_DELAY = 0.5

# Progress is reported every this many feeds. A full pass takes roughly ninety
# seconds, and without periodic output an interactive run is indistinguishable
# from a hung one.
PROGRESS_EVERY = 10

# The endpoint returns 403 to clients that do not present a browser User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def fetch_yahoo_news(ticker):
    """Fetch and parse one ticker's RSS feed.

    Args:
        ticker: Symbol whose feed should be retrieved.

    Returns:
        A list of dicts with keys ticker, headline, published_at and link.

    Raises:
        requests.RequestException: On transport failure or a non-2xx response.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

    # Without this check an error page would be handed to the parser, which finds
    # no items and reports an empty feed rather than a failed request.
    response.raise_for_status()

    soup = BeautifulSoup(response.content, features="xml")

    news_data = []
    for article in soup.find_all("item"):
        # An item without a link cannot be deduplicated, and one without a
        # publication date cannot be assigned to a trading session. Both are
        # discarded rather than stored with placeholder values.
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
    """Scrape every configured ticker and persist the results.

    Raises:
        SystemExit: If no feed yielded any item. A scheduler observes only the
            exit status, so an empty collection must not be reported as success.
    """
    all_news = []
    failed = []

    # Individual successes are not logged: at this universe size a line per feed
    # would add well over a thousand lines a day to pipeline.log and bury the
    # summary. Failures and a periodic counter are logged instead.
    print(f"Fetching news for {len(TICKERS)} tickers...")
    for index, ticker in enumerate(TICKERS):
        if index:
            time.sleep(REQUEST_DELAY)

        try:
            all_news.extend(fetch_yahoo_news(ticker))
        except requests.RequestException as exc:
            # Failures are isolated per feed so that one unavailable ticker does
            # not abort a scheduled run.
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

    # Collecting nothing at all indicates a systemic fault -- blocked requests,
    # lost connectivity, or a changed feed format -- rather than the ordinary
    # case of a feed repeating items already stored. The two conditions are
    # distinguished because they call for different diagnoses.
    if not all_news:
        if len(failed) == len(TICKERS):
            raise SystemExit(f"FAILED: all {len(TICKERS)} feeds unreachable")
        raise SystemExit("FAILED: feeds responded but no items were parsed")


if __name__ == "__main__":
    run()
