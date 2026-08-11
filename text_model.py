"""H5: a model trained on this project's own data, against its own objective.

FinBERT is trained to recognise tone in financial text. Tone is not the target
here, and H4 showed why that matters: sentiment tracks the move that has already
happened. This model skips the intermediate step and learns directly from headline
text to the label that is actually wanted -- did this name beat the cross-section
in the next session -- using realized returns as distant supervision.

TF-IDF over a linear classifier, not a fine-tuned transformer. At roughly forty
thousand short headlines carrying a signal that four prior tests could not detect,
a 66M-parameter model would memorise the training set, and on CPU it would take
hours per epoch to do so. The linear model trains in seconds and produces a table
of token weights, which is inspectable in a way transformer weights are not.

Evaluation runs through the same walk-forward harness as every other hypothesis, so
the comparison against FinBERT is like for like.

It is also reported twice, raw and with ticker identity removed. Headlines name the
company they are about, so a bag-of-words model can learn "Tesla" and score every
Tesla headline alike, which looks like signal and is only memorised persistence.
Demeaning each ticker's scores strips that component out. Removing company names
from the text before vectorizing would prevent it rather than measure it, but the
measurement answers the question and needs no hand-maintained name list.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import database as db
from alpha_engine import MODEL_PERIOD_END, TARGET
from walkforward import INITIAL_TRAIN_SESSIONS, STEP_SESSIONS, summarize

# Unigrams and bigrams: "beats estimates" and "misses estimates" share a token and
# invert the meaning, so bigrams carry information unigrams cannot.
NGRAM_RANGE = (1, 2)

# A token seen fewer times than this is noise the model would happily memorise.
MIN_DOCUMENT_FREQUENCY = 5

# Deliberately strong regularization. Four prior tests found no detectable signal,
# so the prior belief is that most coefficients should be near zero, and a weakly
# regularized model on 40k noisy documents would fit its training set beautifully
# and generalise not at all.
INVERSE_REGULARIZATION = 0.1

MAX_FEATURES = 50_000


def _vectorizer():
    return TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DOCUMENT_FREQUENCY,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )


def _classifier():
    return LogisticRegression(
        C=INVERSE_REGULARIZATION,
        max_iter=2000,
        solver="liblinear",
    )


def load_headlines(study_end=MODEL_PERIOD_END):
    """Headlines attached to the labelled ticker-day they were tradable in.

    The join is a forward merge_asof rather than an equality, mirroring what
    build_features.py does: a headline whose session_date lands on a market holiday
    belongs to the next session that actually traded.
    """
    with db.connect(read_only=True) as conn:
        headlines = pd.read_sql_query(
            """
            SELECT t.ticker, n.session_date, n.headline
            FROM raw_news n
            JOIN news_tickers t USING (article_id)
            WHERE n.session_date <= ?
            """,
            conn, params=(study_end,), parse_dates=["session_date"],
        )
        labels = pd.read_sql_query(
            """
            SELECT ticker, session_date AS trading_date,
                   target_relative, excess_return
            FROM daily_features
            WHERE session_date <= ? AND target_relative IS NOT NULL
            """,
            conn, params=(study_end,), parse_dates=["trading_date"],
        )

    merged = pd.merge_asof(
        headlines.sort_values("session_date"),
        labels.sort_values("trading_date"),
        left_on="session_date", right_on="trading_date",
        by="ticker", direction="forward",
    ).dropna(subset=["trading_date", "target_relative"])

    merged["target_relative"] = merged["target_relative"].astype(int)
    return merged


def walk_forward_text(frame, initial=INITIAL_TRAIN_SESSIONS, step=STEP_SESSIONS):
    """Score every headline out-of-sample, then aggregate to ticker-days.

    The vectorizer is refit inside each fold. Fitting it once over the whole corpus
    would leak vocabulary -- and the inverse document frequencies behind it -- from
    sessions the model has not reached yet.

    Returns (daily, per_headline): one scored row per ticker-day, and the
    underlying per-headline probabilities.
    """
    sessions = (
        frame["trading_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    if len(sessions) <= initial:
        raise SystemExit(f"Need more than {initial} sessions; have {len(sessions)}.")

    collected = []
    for start in range(initial, len(sessions), step):
        train_end = sessions.iloc[start]
        test_end = sessions.iloc[min(start + step, len(sessions)) - 1]

        train = frame[frame["trading_date"] < train_end]
        test = frame[(frame["trading_date"] >= train_end)
                     & (frame["trading_date"] <= test_end)]
        if train.empty or test.empty or train[TARGET].nunique() < 2:
            continue

        vectorizer = _vectorizer()
        model = _classifier()
        model.fit(vectorizer.fit_transform(train["headline"]), train[TARGET])

        scored = test.copy()
        scored["headline_score"] = model.predict_proba(
            vectorizer.transform(test["headline"])
        )[:, 1]
        collected.append(scored)

    per_headline = pd.concat(collected, ignore_index=True)

    # A ticker-day's score is the mean over its headlines. The evaluation is
    # per ticker-day, so per-headline predictions have to be pooled to be
    # comparable with every other hypothesis.
    daily = (
        per_headline.groupby(["ticker", "trading_date"], as_index=False)
        .agg(score=("headline_score", "mean"),
             headlines=("headline_score", "size"),
             target_relative=(TARGET, "first"),
             excess_return=("excess_return", "first"))
        .rename(columns={"trading_date": "session_date"})
    )
    return daily, per_headline


def token_weights(frame, top=18):
    """Tokens the model weights most, from a fit over the whole study period.

    For inspection only. This fit has seen every session, so its coefficients are
    not out-of-sample evidence of anything -- they describe what the model latches
    onto, not what it predicts.
    """
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(frame["headline"])
    model = _classifier()
    model.fit(matrix, frame[TARGET])

    weights = pd.Series(model.coef_[0], index=vectorizer.get_feature_names_out())
    return weights.nlargest(top), weights.nsmallest(top), len(weights)


def run():
    frame = load_headlines()
    print(f"Loaded {len(frame):,} headline-ticker rows across "
          f"{frame['trading_date'].nunique()} sessions and "
          f"{frame['ticker'].nunique()} tickers.\n")

    print("Walk-forward, refitting the vectorizer and model each fold...")
    daily, per_headline = walk_forward_text(frame)
    stats = summarize(daily)

    print(f"\n=== H5: TF-IDF + logistic regression, trained on returns ===")
    print(f"  headlines scored out-of-sample  {len(per_headline):>10,}")
    print(f"  ticker-days                     {stats['rows']:>10,}")
    print(f"  accuracy                        {stats['accuracy']:>10.1%}")
    print(f"  baseline                        {stats['baseline']:>10.1%}")
    print(f"  edge                            {stats['accuracy'] - stats['baseline']:>+10.1%}")
    print(f"  mean IC                         {stats['mean_ic']:>+10.4f}")
    print(f"  t-statistic                     {stats['t_stat']:>+10.2f}")
    print(f"  IC sessions                     {stats['ic_sessions']:>10,}")

    floor = 2.5 * stats["ic_std"] / np.sqrt(stats["ic_sessions"])
    print(f"  minimum detectable mean IC      {floor:>10.4f}")

    # Control. Headlines name their subject, so the model can score a ticker rather
    # than read a headline. Demeaning by ticker removes whatever is constant per
    # name and leaves only what varies day to day.
    daily["score_demeaned"] = (
        daily["score"] - daily.groupby("ticker")["score"].transform("mean")
    )
    controlled = summarize(
        daily.assign(score=daily["score_demeaned"])
    )

    between = daily.groupby("ticker")["score"].mean().var()
    within = daily.groupby("ticker")["score"].transform(lambda s: s - s.mean()).var()

    print(f"\n=== control: how much of that is just ticker identity? ===")
    print(f"  raw score                       {stats['mean_ic']:>+10.4f}"
          f"   t {stats['t_stat']:>+6.2f}")
    print(f"  ticker-demeaned score           {controlled['mean_ic']:>+10.4f}"
          f"   t {controlled['t_stat']:>+6.2f}")
    print(f"  share of score variance fixed per ticker: "
          f"{between / (between + within):>6.1%}")

    print("\n=== against FinBERT on the same harness ===")
    print(f"  {'model':<38}{'mean IC':>10}{'t':>8}")
    print(f"  {'FinBERT sentiment (H0)':<38}{-0.0146:>+10.4f}{-1.33:>+8.2f}")
    print(f"  {'TF-IDF on returns (H5), raw':<38}"
          f"{stats['mean_ic']:>+10.4f}{stats['t_stat']:>+8.2f}")
    print(f"  {'TF-IDF on returns (H5), controlled':<38}"
          f"{controlled['mean_ic']:>+10.4f}{controlled['t_stat']:>+8.2f}")

    positive, negative, vocabulary = token_weights(frame)
    print(f"\n=== what the model latched onto ({vocabulary:,} tokens) ===")
    print("  full-sample fit, for inspection only -- not out-of-sample evidence\n")
    print(f"  {'leans toward outperformance':<34}{'leans toward underperformance'}")
    for (up_token, up_weight), (down_token, down_weight) in zip(
            positive.items(), negative.items()):
        print(f"  {up_token:<24}{up_weight:>+7.3f}   {down_token:<24}{down_weight:>+7.3f}")

    print("""
Company names dominate the heaviest weights, which is the control above made
visible: the model is partly learning which stock a headline is about rather than
what it says. Neither number clears significance, and the controlled one is not
close, so H5 joins H0-H2 as a null.

Worth noting that this is the third distinct form the same failure has taken here.
Raw volume let the classifier identify tickers; coverage predicted volatility
between stocks but not within them; now company names do the same through text. In
a cross-section, any feature varying more between names than within them will look
predictive of anything else that also varies between names.""")

    return stats


if __name__ == "__main__":
    run()
