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
from scraper import UNIVERSE

# Long enough that clicking around does not re-query, short enough that an hourly
# scrape appears without a manual refresh.
CACHE_TTL = 300

SECTOR_OF = {ticker: sector for sector, tickers in UNIVERSE.items()
             for ticker in tickers}

# A ticker needs at least this many scored articles in the window to appear on the
# leaderboard. One stray headline is not a read on a company.
MIN_ARTICLES_FOR_RANKING = 3

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
def load_recent_sentiment(days=3, min_articles=MIN_ARTICLES_FOR_RANKING):
    """Per-ticker sentiment over the most recent sessions, with sector attached.

    Anchored to the newest session in the database rather than to today's date, so
    the view still reads correctly over a weekend or after the scraper has been
    down.
    """
    with db.connect(read_only=True) as conn:
        latest = conn.execute("SELECT MAX(session_date) FROM raw_news").fetchone()[0]
        if latest is None:
            return pd.DataFrame(), None

        cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=days - 1)).date().isoformat()
        frame = pd.read_sql_query(
            """
            SELECT t.ticker,
                   COUNT(*)                                      AS articles,
                   AVG(s.net_sentiment)                          AS avg_net,
                   SUM(s.sentiment_label = 'positive')           AS positive,
                   SUM(s.sentiment_label = 'negative')           AS negative
            FROM raw_news n
            JOIN news_tickers t USING (article_id)
            JOIN sentiment_scores s USING (article_id)
            WHERE n.session_date >= ?
            GROUP BY t.ticker
            HAVING COUNT(*) >= ?
            """,
            conn,
            params=(cutoff, min_articles),
        )

    frame["sector"] = frame["ticker"].map(SECTOR_OF)
    return frame, latest


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


def _diverging_bars(frame, value, label, title, height):
    """Horizontal bars sorted by value, red below zero and green above."""
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title="Average sentiment  (-1 to +1)"),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum[value] > 0,
                alt.value(SENTIMENT_COLORS["positive"]),
                alt.value(SENTIMENT_COLORS["negative"]),
            ),
            tooltip=[f"{label}:N", alt.Tooltip(f"{value}:Q", format="+.3f"),
                     "articles:Q"],
        )
        .properties(title=title, height=height)
    )


def render_today():
    st.markdown(
        "What the news is saying right now. Each stock's score is the average "
        "sentiment of its recent headlines, from **−1 (all negative)** to "
        "**+1 (all positive)**."
    )

    window = st.radio("Window", [3, 7, 14], index=0, horizontal=True,
                      format_func=lambda days: f"Last {days} sessions")
    frame, latest = load_recent_sentiment(days=window)

    if frame.empty:
        st.info("No scored headlines in this window. Run sentiment_analyzer.py.")
        return

    st.caption(
        f"Most recent session in the database: **{latest}**. "
        f"{len(frame)} tickers with at least {MIN_ARTICLES_FOR_RANKING} scored "
        f"articles, {int(frame['articles'].sum()):,} articles in total. "
        "Articles collected but not yet scored do not appear."
    )

    columns = st.columns(3)
    most_positive = frame.loc[frame["avg_net"].idxmax()]
    most_negative = frame.loc[frame["avg_net"].idxmin()]
    columns[0].metric("Most positive", most_positive["ticker"],
                      f"{most_positive['avg_net']:+.3f}")
    columns[1].metric("Most negative", most_negative["ticker"],
                      f"{most_negative['avg_net']:+.3f}")
    columns[2].metric("Universe average", f"{frame['avg_net'].mean():+.3f}",
                      f"{int(frame['articles'].sum()):,} articles")

    # Sectors first: eight bars are readable at a glance where fifty-seven are not.
    # Weighted by article count so a thinly covered name cannot swing a sector.
    sectors = (
        frame.dropna(subset=["sector"])
        .assign(weighted=lambda d: d["avg_net"] * d["articles"])
        .groupby("sector", as_index=False)
        .agg(weighted=("weighted", "sum"), articles=("articles", "sum"))
    )
    sectors["avg_net"] = sectors["weighted"] / sectors["articles"]
    st.altair_chart(
        _diverging_bars(sectors, "avg_net", "sector", "By sector", 260),
        use_container_width=True,
    )

    extremes = st.slider("Stocks to show at each end", 5, 25, 10)
    ranked = pd.concat([
        frame.nlargest(extremes, "avg_net"),
        frame.nsmallest(extremes, "avg_net"),
    ]).drop_duplicates(subset="ticker")
    st.altair_chart(
        _diverging_bars(ranked, "avg_net", "ticker",
                        f"Most positive and most negative {extremes}",
                        max(300, 22 * len(ranked))),
        use_container_width=True,
    )

    with st.expander("All tickers, as a table"):
        table = frame.sort_values("avg_net", ascending=False)[
            ["ticker", "sector", "articles", "positive", "negative", "avg_net"]
        ]
        st.dataframe(table.round({"avg_net": 3}), hide_index=True,
                     use_container_width=True)


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
    st.subheader("In plain language")
    st.markdown(
        """
        The obvious idea is that good news should mean the stock goes up tomorrow.
        Tested properly, **it doesn't** — at least not in a way you could trade.

        Two follow-up questions explain why, and both have clearer answers:

        - **Sentiment describes the present, not the future.** A headline's tone
          tracks the move that has *already happened*. By the time it is published,
          the price has moved.
        - **How much a company is written about does say something about how
          volatile it will be** — not which direction, just how big the swings.

        So the news is informative about *risk*, and roughly useless about
        *direction*. The detail behind each claim is below.
        """
    )

    st.divider()
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

    st.divider()
    st.subheader("What the same data does relate to")
    st.markdown(
        """
        **H4 — sentiment is contemporaneous, not predictive.** The same feature,
        the same sample, correlated against the past and against the future:

        | Sentiment correlated against | Mean IC | t |
        |---|---|---|
        | the return that already happened | +0.0255 | **+2.42** |
        | the next session's excess return | −0.0023 | −0.23 |

        That pair is the explanation for the null. Headlines report moves rather
        than anticipating them, which is a mechanical consequence of how they are
        written.

        **H3 — coverage relates to forward volatility, in the cross-section only.**

        | Test | Mean IC | t |
        |---|---|---|
        | coverage level → \\|next return\\| (uncontrolled) | +0.0469 | +4.91 |
        | trailing volatility → \\|next return\\| | +0.2130 | +19.28 |
        | coverage → volatility residual *(controlled)* | +0.0281 | **+3.13** |
        | within-ticker coverage → within-ticker \\|return\\| | −0.0012 | −0.14 |

        The uncontrolled figure overstates it: volatility clusters, so anything
        correlated with a stock being volatile looks predictive. It survives that
        control at t = +3.13, but the within-ticker test is flat. So coverage is a
        **cross-sectional risk characteristic** — widely covered names are riskier
        names — and **not a timing signal**. A given stock's busy news day says
        nothing about that stock's next session.
        """
    )

    st.divider()
    st.subheader("A model trained on this project's own data")
    st.markdown(
        """
        FinBERT is trained to recognise tone, and H4 shows tone is contemporaneous.
        **H5** skips that step: TF-IDF over headline text into a linear classifier,
        trained directly on whether the name beat the cross-section next session,
        run through the same walk-forward harness.

        | Model | Mean IC | t |
        |---|---|---|
        | FinBERT sentiment (H0) | −0.0146 | −1.33 |
        | TF-IDF on returns (H5), raw | +0.0231 | +1.90 |
        | TF-IDF on returns (H5), controlled | +0.0096 | +0.83 |

        The raw figure looks like an improvement and mostly is not. Headlines name
        the company they describe, so a bag-of-words model can learn *Tesla* and
        score every Tesla headline alike — the heaviest weights are
        `tesla +0.62`, `ford −0.74`, `oracle +0.47`. **13.7%** of the score is a
        fixed offset per ticker, and demeaning it removes over half the edge.
        Neither figure is significant.

        That is the third form the same failure took here: raw volume, then
        coverage, now company names. In a cross-section, a feature that varies more
        *between* names than *within* them will look predictive of anything else
        that also varies between names.
        """
    )

    st.divider()
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

    # "Today" leads because it is the only tab that answers the question a general
    # visitor actually arrives with.
    today, feed, chart, pipeline, research = st.tabs(
        ["Today", "Live feed", "Sentiment vs price", "Pipeline", "Research findings"]
    )
    with today:
        render_today()
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
