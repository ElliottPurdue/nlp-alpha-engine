"""Streamlit dashboard.

Opens the database read-only, which WAL mode makes safe while a scheduled scrape is
writing. Without WAL a run in progress would block every query here.

The live sections and the research section are kept apart on purpose. Showing a
sentiment feed without the finding next to it would imply a working strategy the
evidence does not support.

Run with:  streamlit run app.py
"""

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import database as db

# Long enough that clicking around does not re-query, short enough that an hourly
# scrape appears without a manual refresh.
CACHE_TTL = 300

SENTIMENT_COLORS = {
    "positive": "#1a7f5a",
    "negative": "#b3261e",
    "neutral": "#6b6b6b",
}

# Divides the FNSPID backfill from live collection. The two have different sources
# and coverage and are never mixed in one view.
LIVE_PERIOD_START = "2026-01-01"

ROLLING_WINDOW = 7

# FNSPID covers JPM on 68% of historical sessions against AAPL's 10%, so it opens on
# a dense series rather than a near-empty one.
DEFAULT_CHART_TICKER = "JPM"

st.set_page_config(page_title="NLP Alpha Engine", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL)
def load_summary():
    with db.connect(read_only=True) as conn:
        counts = db.table_counts(conn)
        row = conn.execute(
            "SELECT MAX(last_seen_at) AS last_scrape,"
            "       COUNT(DISTINCT session_date) AS sessions FROM raw_news"
        ).fetchone()
        tickers = len(db.distinct_tickers(conn))
    return counts, row["last_scrape"], row["sessions"], tickers


@st.cache_data(ttl=CACHE_TTL)
def load_recent_headlines(limit=400):
    with db.connect(read_only=True) as conn:
        return pd.read_sql_query(
            """
            SELECT ticker, headline, url, source, published_at, session_date,
                   sentiment_label, sentiment_score, net_sentiment
            FROM v_scored_headlines
            WHERE sentiment_label IS NOT NULL
            ORDER BY published_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


@st.cache_data(ttl=CACHE_TTL)
def live_window_start(buffer_days=21):
    """Start of the live chart window, taken from the data.

    Live collection is weeks old while price history runs ten years, so a fixed
    start date would draw hundreds of price points beside a handful of sentiment
    points.
    """
    with db.connect(read_only=True) as conn:
        earliest = conn.execute(
            "SELECT MIN(session_date) FROM daily_features WHERE session_date >= ?",
            (LIVE_PERIOD_START,),
        ).fetchone()[0]
    if earliest is None:
        return LIVE_PERIOD_START
    return (pd.Timestamp(earliest) - pd.Timedelta(days=buffer_days)).date().isoformat()


@st.cache_data(ttl=CACHE_TTL)
def load_ticker_series(ticker, start, end):
    """Prices for one ticker with sentiment attached where it exists.

    Driven from `prices` so the price line stays continuous through sessions with no
    news. Bounded at both ends because sentiment exists in two disjoint blocks and an
    open window would stretch the chart across the six-year gap between them.
    """
    with db.connect(read_only=True) as conn:
        frame = pd.read_sql_query(
            """
            SELECT p.date, p.close, f.mean_sentiment, f.headline_count
            FROM prices p
            LEFT JOIN daily_features f
                   ON f.ticker = p.ticker AND f.session_date = p.date
            WHERE p.ticker = ? AND p.date >= ? AND p.date <= ?
            ORDER BY p.date
            """,
            conn,
            params=(ticker, start, end),
            parse_dates=["date"],
        )
    frame["rolling_sentiment"] = (
        frame["mean_sentiment"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return frame


@st.cache_data(ttl=CACHE_TTL)
def load_coverage():
    with db.connect(read_only=True) as conn:
        return pd.read_sql_query(
            """
            SELECT session_date, COUNT(*) AS articles
            FROM raw_news GROUP BY session_date ORDER BY session_date
            """,
            conn,
            parse_dates=["session_date"],
        )


@st.cache_data(ttl=CACHE_TTL)
def load_universe():
    with db.connect(read_only=True) as conn:
        return db.distinct_tickers(conn)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_headline_card(row):
    """One headline with a sentiment-coloured left border.

    Headline and source are escaped: they are third-party RSS text, and pasting them
    unescaped would both break the layout and inject arbitrary HTML.
    """
    colour = SENTIMENT_COLORS.get(row.sentiment_label, "#6b6b6b")
    st.markdown(
        f"""
        <div style="border-left:4px solid {colour};padding:0.35rem 0 0.35rem 0.75rem;
                    margin-bottom:0.5rem;">
          <div style="font-size:0.78rem;color:#888;">
            <strong>{html.escape(row.ticker)}</strong>
            &nbsp;·&nbsp; {html.escape(str(row.source))}
            &nbsp;·&nbsp; {html.escape(str(row.published_at))}
            &nbsp;·&nbsp; <span style="color:{colour};">
              {html.escape(row.sentiment_label)} {row.sentiment_score:.2f}
            </span>
          </div>
          <div style="font-size:0.95rem;">{html.escape(row.headline)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_feed():
    headlines = load_recent_headlines()
    if headlines.empty:
        st.info("No scored headlines yet. Run scraper.py then sentiment_analyzer.py.")
        return

    universe = ["All tickers"] + load_universe()
    chosen = st.selectbox("Ticker", universe, key="feed_ticker")
    labels = st.multiselect(
        "Sentiment", ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )

    view = headlines[headlines["sentiment_label"].isin(labels)]
    if chosen != "All tickers":
        view = view[view["ticker"] == chosen]

    st.caption(f"{len(view)} of the {len(headlines)} most recent scored headlines")
    for row in view.head(60).itertuples():
        render_headline_card(row)


def render_sentiment_vs_price():
    universe = load_universe()
    default = universe.index(DEFAULT_CHART_TICKER) if DEFAULT_CHART_TICKER in universe else 0
    ticker = st.selectbox("Ticker", universe, index=default, key="chart_ticker")

    # Both windows are bounded to where sentiment exists. The historical block is
    # dense and opens by default; live collection is a fortnight old and fills in as
    # the scheduled scrape keeps running.
    periods = {
        "Historical backfill (2018-2020)": ("2018-01-01", "2020-06-30"),
        "Live collection": (live_window_start(), "2100-01-01"),
    }
    label = st.radio("Period", list(periods), horizontal=True)
    start, end = periods[label]

    frame = load_ticker_series(ticker, start, end)
    observed = int(frame["mean_sentiment"].notna().sum())
    if frame.empty or observed == 0:
        st.info(f"No sentiment data for {ticker} in this period.")
        return

    base = alt.Chart(frame).encode(x=alt.X("date:T", title=None))
    price_line = base.mark_line(color="#4c78a8", strokeWidth=1.8).encode(
        y=alt.Y("close:Q", title="Close price", scale=alt.Scale(zero=False)),
        tooltip=["date:T", "close:Q", "headline_count:Q"],
    )
    sentiment_line = base.mark_line(color="#e45756", strokeWidth=1.6).encode(
        y=alt.Y("rolling_sentiment:Q",
                title=f"{ROLLING_WINDOW}-day mean sentiment"),
    )

    # Independent scales: sentiment sits in [-1, 1] and price in the hundreds, so a
    # shared axis flattens the sentiment line to nothing.
    st.altair_chart(
        alt.layer(price_line, sentiment_line).resolve_scale(y="independent")
        .properties(height=420),
        use_container_width=True,
    )
    st.caption(
        f"Blue: close price (left axis). Red: {ROLLING_WINDOW}-day rolling mean "
        f"sentiment (right axis). {observed} sentiment observations across "
        f"{len(frame)} trading sessions — sentiment exists only on days the "
        f"ticker had news."
    )


def render_pipeline():
    coverage = load_coverage()
    st.subheader("Articles collected per trading session")
    st.altair_chart(
        alt.Chart(coverage).mark_bar(color="#4c78a8").encode(
            x=alt.X("session_date:T", title=None),
            y=alt.Y("articles:Q", title="Articles"),
            tooltip=["session_date:T", "articles:Q"],
        ).properties(height=260),
        use_container_width=True,
    )
    st.caption(
        "The gap is real: the FNSPID backfill ends June 2020 and live collection "
        "began in 2026. The two blocks come from different sources with different "
        "coverage, so the study is fitted on the historical block and the live "
        "period is reserved as an untouched forward test."
    )

    live = coverage[coverage["session_date"] >= LIVE_PERIOD_START]
    historical = coverage[coverage["session_date"] < LIVE_PERIOD_START]
    left, right = st.columns(2)
    left.metric("Historical sessions (backfill)", f"{len(historical):,}",
                f"{int(historical['articles'].sum()):,} articles")
    right.metric("Live sessions", f"{len(live):,}",
                 f"{int(live['articles'].sum()):,} articles")


def render_research():
    st.subheader("Does headline sentiment predict relative performance?")
    st.markdown(
        """
        **No detectable signal.** Evaluated walk-forward with monthly refits over
        365 out-of-sample sessions (2018–2020, 47 large-cap US equities), the mean
        information coefficient is **−0.015 with t = −1.33**. Five pre-registered
        configurations were tested and all five are reported; none reaches the
        Bonferroni-corrected threshold of |t| ≥ 2.5.

        The sample could only have detected a mean IC above **0.027**. Published
        daily equity news signals typically run 0.01–0.03, so this rules out a
        large effect, not a small one.
        """
    )

    st.dataframe(
        pd.DataFrame([
            ("H0  level / 1-day", 9046, "48.9%", "-2.5%", -0.0146, 365, -1.33),
            ("H1  surprise / 1-day", 8868, "49.9%", "-1.4%", 0.0010, 359, 0.10),
            ("H2  level / 5-day", 9046, "49.9%", "-1.4%", -0.0013, 73, -0.05),
            ("H1+H2  surprise / 5-day", 8868, "49.1%", "-2.1%", 0.0120, 72, 0.45),
            ("aux  combined / 1-day", 8868, "49.4%", "-1.9%", -0.0124, 359, -1.08),
        ], columns=["configuration", "rows", "accuracy", "edge vs baseline",
                    "mean IC", "IC sessions", "t"]),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Why the backtest is more misleading than the IC")
    curve = Path(__file__).resolve().parent / "equity_curve.png"
    if curve.exists():
        st.image(str(curve), use_container_width=True)
    else:
        st.info("Run backtest.py to generate equity_curve.png.")

    st.markdown(
        """
        A dollar-neutral quintile long/short book rebalanced daily returns
        **+7.8% annualized gross, Sharpe +0.50** — which looks like an edge and is
        not one. The entire gross return comes from Q2 2020: excluding that single
        quarter it is **−4.7%**, and two of seven quarters are outright negative.

        Turnover of **3.36× capital per session** implies a **42% annual cost
        drag** at 5bps, leaving **−29.4% annualized net**. Holding the universe
        equal-weight returned +18.0% at a higher Sharpe than the strategy's gross.

        The information coefficient weights every session equally; profit and loss
        weights by magnitude. That is why a handful of violent sessions produced an
        apparently positive backtest while the IC correctly reported nothing.
        """
    )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def main():
    st.title("NLP-Driven Financial Sentiment & Alpha Engine")

    counts, last_scrape, sessions, tickers = load_summary()
    columns = st.columns(5)
    columns[0].metric("Articles", f"{counts['raw_news']:,}")
    columns[1].metric("Scored", f"{counts['sentiment_scores']:,}")
    columns[2].metric("Tickers", tickers)
    columns[3].metric("Sessions", f"{sessions:,}")
    columns[4].metric("Price bars", f"{counts['prices']:,}")
    st.caption(f"Most recent scrape: {last_scrape or 'never'} · "
               f"database opened read-only, safe to view mid-pipeline")

    feed, chart, pipeline, research = st.tabs(
        ["Live feed", "Sentiment vs price", "Pipeline", "Research findings"]
    )
    with feed:
        render_live_feed()
    with chart:
        render_sentiment_vs_price()
    with pipeline:
        render_pipeline()
    with research:
        render_research()


if __name__ == "__main__":
    main()
