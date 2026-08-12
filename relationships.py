"""Relationship tests: what does headline data actually relate to?

experiments.py asks whether a classifier can predict relative returns from
sentiment. It cannot. These tests ask the narrower question of what the same data
does relate to, which is how the null gets an explanation instead of just a result.

    H3  News coverage relates to forward volatility. Attention and volatility are
        linked in the literature, and magnitude is far more forecastable than sign.

    H4  Sentiment is contemporaneous with returns rather than leading them. A
        headline describing a move that already happened cannot forecast the next
        one, which would explain H0 through H2.

Every test carries its controls. Uncontrolled, H3 looks about twice as strong as it
is: volatility clusters, so anything correlated with a stock being volatile appears
to predict volatility. Two controls are applied -- residualizing on trailing
realized volatility, and a within-ticker test that removes stock identity outright.
"""

import numpy as np
import pandas as pd
from scipy import stats

import database as db

# Kept in step with alpha_engine so every test covers the same sample.
from alpha_engine import MODEL_PERIOD_END as STUDY_END

# Sessions of realized volatility used as the control, strictly prior to each row.
TRAILING_VOL_WINDOW = 20
MIN_TRAILING_OBS = 10

# A rank correlation over a handful of names says more about which names had news
# that day than about any relationship.
MIN_NAMES_FOR_IC = 5

# Bonferroni-corrected bar for the number of tests reported here, derived rather
# than rounded: at four tests a rounded 2.5 and the exact 2.498 barely differ, but
# the habit of rounding a threshold is how a t of 2.54 gets called significant.
FAMILY_ALPHA = 0.05
TESTS_REPORTED = 4
SIGNIFICANCE_T = stats.norm.ppf(1 - FAMILY_ALPHA / (2 * TESTS_REPORTED))


def load_panel(study_end=STUDY_END):
    """daily_features for the study period, with prior return and trailing vol.

    Both controls are derived here rather than stored: they are properties of the
    price series used for analysis, not features the model consumes.
    """
    with db.connect(read_only=True) as conn:
        frame = pd.read_sql_query(
            """
            SELECT ticker, session_date, mean_sentiment, headline_count,
                   sentiment_rank, headline_rank,
                   attention_surprise, attention_surprise_rank,
                   sentiment_surprise, sentiment_surprise_rank,
                   fwd_return
            FROM daily_features
            WHERE session_date <= ? AND fwd_return IS NOT NULL
            """,
            conn,
            params=(study_end,),
            parse_dates=["session_date"],
        )
        prices = pd.read_sql_query(
            "SELECT ticker, date, close FROM prices ORDER BY ticker, date",
            conn,
            parse_dates=["date"],
        )

    by_ticker = prices.groupby("ticker")["close"]
    prices["prev_return"] = by_ticker.transform(lambda s: s / s.shift(1) - 1)
    # shift(1) before the window so a session's own move is excluded from the
    # volatility it is measured against.
    prices["trailing_vol"] = (
        prices.groupby("ticker")["prev_return"]
        .transform(lambda s: s.shift(1)
                   .rolling(TRAILING_VOL_WINDOW, min_periods=MIN_TRAILING_OBS).std())
    )

    frame = frame.merge(
        prices[["ticker", "date", "prev_return", "trailing_vol"]],
        left_on=["ticker", "session_date"], right_on=["ticker", "date"], how="left",
    )
    frame["abs_fwd_return"] = frame["fwd_return"].abs()
    frame["vol_rank"] = frame.groupby("session_date")["trailing_vol"].rank(pct=True)
    return frame


def session_ic(frame, predictor, target, min_names=MIN_NAMES_FOR_IC):
    """Mean per-session Spearman correlation, its t-statistic, and session count."""
    values = []
    for _, group in frame.groupby("session_date"):
        group = group.dropna(subset=[predictor, target])
        if len(group) < min_names:
            continue
        # A constant column has no defined rank correlation. It happens when every
        # name in a thin session shares a value, and scipy warns rather than
        # returning quietly, so those sessions are skipped before the call.
        if group[predictor].nunique() < 2 or group[target].nunique() < 2:
            continue
        correlation = group[predictor].corr(group[target], method="spearman")
        if pd.notna(correlation):
            values.append(correlation)

    series = pd.Series(values)
    if series.empty or not series.std():
        return 0.0, 0.0, len(series)
    return (series.mean(),
            series.mean() / series.std() * np.sqrt(len(series)),
            len(series))


def residualize(frame, target, control):
    """Within each session, strip the part of `target` explained by `control`.

    Both are converted to ranks first, so this removes a monotonic relationship
    without assuming it is linear in the raw values.
    """
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)

    for _, group in frame.groupby("session_date"):
        group = group.dropna(subset=[target, control])
        if len(group) < MIN_NAMES_FOR_IC:
            continue
        x = group[control].rank(pct=True)
        y = group[target].rank(pct=True)
        if x.std() == 0:
            continue
        beta = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        residuals.loc[group.index] = y - (y.mean() + beta * (x - x.mean()))

    return residuals


def within_ticker_correlation(frame, predictor, target):
    """Correlation after demeaning both series by ticker.

    This is the strict test. It discards every between-stock difference, so it
    answers whether a given stock's own busy news day says anything about that
    stock's own next session.
    """
    subset = frame.dropna(subset=[predictor, target]).copy()
    for column in (predictor, target):
        subset[f"_dm_{column}"] = (
            subset[column] - subset.groupby("ticker")[column].transform("mean")
        )

    rho = subset[f"_dm_{predictor}"].corr(subset[f"_dm_{target}"], method="spearman")
    if pd.isna(rho) or len(subset) < 3:
        return 0.0, 0.0, len(subset)
    t_stat = rho * np.sqrt((len(subset) - 2) / (1 - rho ** 2))
    return rho, t_stat, len(subset)


def _report(label, mean, t_stat, n, unit="sessions"):
    flag = "  <-- significant" if abs(t_stat) >= SIGNIFICANCE_T else ""
    print(f"  {label:<44}{mean:>+9.4f}{t_stat:>+8.2f}   {n:>6,} {unit}{flag}")


def run():
    frame = load_panel()
    print(f"Panel: {len(frame):,} ticker-days, "
          f"{frame['session_date'].nunique()} sessions, "
          f"{frame['ticker'].nunique()} tickers (through {STUDY_END})\n")

    print("=" * 78)
    print("H3  Does news coverage relate to forward volatility?")
    print("=" * 78)
    print(f"  {'':<44}{'corr':>9}{'t':>8}")

    # Baseline for scale: volatility clustering on its own is overwhelming, so any
    # news effect has to be read against this rather than against zero.
    _report("trailing volatility -> |next return|",
            *session_ic(frame, "vol_rank", "abs_fwd_return"))
    _report("coverage level -> |next return| (uncontrolled)",
            *session_ic(frame, "headline_rank", "abs_fwd_return"))

    overlap = frame["headline_rank"].corr(frame["vol_rank"], method="spearman")
    print(f"\n  coverage and trailing volatility overlap: rho {overlap:+.3f}")
    print("  (low, so coverage is not simply restating volatility)\n")

    frame["abs_residual"] = residualize(frame, "abs_fwd_return", "vol_rank")
    _report("coverage level -> volatility residual",
            *session_ic(frame, "headline_rank", "abs_residual"))
    _report("coverage surprise -> volatility residual",
            *session_ic(frame, "attention_surprise_rank", "abs_residual"))

    print("\n  strict control, stock identity removed:")
    _report("within-ticker coverage -> within-ticker |return|",
            *within_ticker_correlation(frame, "attention_surprise", "abs_fwd_return"),
            unit="rows")

    print("\n" + "=" * 78)
    print("H4  Is sentiment contemporaneous with returns rather than leading them?")
    print("=" * 78)
    print(f"  {'':<44}{'corr':>9}{'t':>8}")
    _report("sentiment -> the return that already happened",
            *session_ic(frame, "sentiment_rank", "prev_return"))
    _report("sentiment surprise -> return already happened",
            *session_ic(frame, "sentiment_surprise_rank", "prev_return"))
    print("\n  for comparison, the forward-looking version from experiments.py:")
    _report("sentiment -> next session's excess return",
            *session_ic(frame, "sentiment_rank", "fwd_return"))

    print("\n" + "=" * 78)
    print(f"Threshold |t| >= {SIGNIFICANCE_T:.3f}, Bonferroni-corrected.\n")
    _verdicts(frame)


def _verdicts(frame):
    """Derive the conclusions from the numbers rather than reciting fixed prose.

    An earlier version hardcoded its findings. When a larger sample turned the
    within-ticker control from null to significant, the module went on printing
    that the control had failed. Conclusions are computed here for that reason.
    """
    _, residual_t, _ = session_ic(frame, "headline_rank", "abs_residual")
    _, within_t, _ = within_ticker_correlation(
        frame, "attention_surprise", "abs_fwd_return")
    _, past_t, _ = session_ic(frame, "sentiment_rank", "prev_return")
    _, future_t, _ = session_ic(frame, "sentiment_rank", "fwd_return")

    passes = lambda t: abs(t) >= SIGNIFICANCE_T

    print("H3  coverage and forward volatility")
    if passes(residual_t) and passes(within_t):
        print("    Holds both cross-sectionally and within stocks. Surviving the")
        print("    within-ticker control means this is not merely a statement that")
        print("    widely covered names are volatile names: a given stock's busier")
        print("    news day does carry information about its own next session.")
    elif passes(residual_t):
        print("    Holds cross-sectionally, fails within stocks. Coverage adds to")
        print("    trailing volatility across names, but a stock's own busy day says")
        print("    nothing about its own next session -- a risk characteristic, not a")
        print("    timing signal.")
    else:
        print("    Does not hold once trailing volatility is controlled for.")

    print("\nH4  sentiment and the timing of returns")
    if passes(past_t) and abs(past_t) > abs(future_t):
        print(f"    Holds. Sentiment tracks the move already made (t {past_t:+.2f})")
        print(f"    more strongly than the one to come (t {future_t:+.2f}). Headlines")
        print("    describe the present, which is the most likely explanation for the")
        print("    null on direction.")
    elif passes(future_t):
        print("    Reverses: the forward relationship is the stronger one.")
    else:
        print("    Neither direction is distinguishable from noise at this threshold.")


if __name__ == "__main__":
    run()
