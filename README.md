# NLP-Driven Financial Sentiment & Alpha Engine

An end-to-end quantitative research pipeline that collects financial news headlines,
scores them with a finance-specific transformer model (**FinBERT**), aligns the
resulting sentiment to tradable market sessions, and tests whether it predicts
cross-sectional equity returns.

Sentiment turns out not to predict direction. Follow-up tests explain why, and find
what the same data does relate to.

---

## 📌 Findings

Five pre-registered hypotheses over **250,853 scored headlines** and **2,775 trading
sessions** (2013–2024, 54 large-cap US equities), evaluated walk-forward.

**1. Headline sentiment does not predict next-session relative returns.** The best
configuration reaches mean IC **+0.0104 (t = +2.48)** against a Bonferroni-corrected
threshold of 2.576, with a minimum detectable effect of 0.0107 — the estimate sits
*at* the edge of what the sample can resolve. No configuration clears the bar.

**2. Sentiment is contemporaneous with returns, not predictive of them** — which
explains the first result. Same feature, same sample:

| Sentiment correlated against | Mean IC | t |
|---|---|---|
| the return that already happened | +0.0238 | **+5.47** |
| the next session's excess return | +0.0057 | +1.33 |

By the time a headline is published, the price has moved.

**3. News coverage predicts forward volatility, and it survives every control.**
Not direction — *magnitude*.

| Test | Corr | t |
|---|---|---|
| trailing volatility → \|next return\| *(baseline for scale)* | +0.2258 | +48.34 |
| coverage → volatility residual *(vol controlled)* | +0.0232 | **+5.66** |
| within-ticker coverage → within-ticker \|return\| | +0.0167 | **+4.50** |

The within-ticker control is the strict one: it discards every between-stock
difference, so this says a *given stock's* busier news day predicts *that stock's*
next-session move. It measured −0.14, +3.62, then +4.50 as the sample grew from 615
to 1,887 to 2,775 sessions — strengthening monotonically with n, with a stable
effect size.

**In short: the news says a lot about risk and almost nothing about direction.**

### What growing the sample settled

An earlier run on 1,887 sessions flagged one configuration as significant at
t = +2.54. Adding 49% more data took it to **t = +2.05** with the effect shrinking
from +0.0284 to +0.0192 — a fluke regressing, exactly as it should. Over the same
expansion, H0's estimate held at +0.0103 → +0.0104 while its t rose +1.96 → +2.48,
and H3's within-ticker control went from null to strongly significant.

Growing the sample is what separated the three.

![Equity curve](equity_curve.png)

A dollar-neutral quintile long/short book, rebalanced daily, over 2,507 sessions:

| | Total | Annualized | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| Long/short, gross | +183.5% | +11.0% | 16.5% | +0.72 | -22.8% |
| Long/short, net of 5bps | -94.9% | -25.8% | 16.5% | **-1.73** | -95.7% |
| Equal-weight universe | +340.0% | +16.1% | 19.1% | **+0.88** | -33.6% |

- **Passively holding the universe beats the strategy even before costs** - Sharpe
  +0.88 against +0.72 gross. A small gross spread exists and is not worth having.
- **Turnover is 3.20x capital per session**, a 40.3% annual cost drag. A
  four-name-per-side book rebalanced daily rotates almost completely each day, and
  costs consume several times the gross spread.
- The return is *not* concentrated: 29 of 41 quarters positive, and excluding the
  best quarter still leaves +94.9% of the +117.8% gross total. That is a persistent
  small spread rather than one lucky episode - and still not tradable.

The information coefficient weights every session equally; profit and loss weights
by magnitude. On an earlier, smaller sample the backtest showed Sharpe +0.50 gross
that came entirely from a single quarter while the IC read zero. The IC was right.
Had only the backtest been built, this project would have reported a false positive.

---

## 🧪 Pre-registered hypotheses

Every hypothesis was stated before its result was computed, and every configuration
is reported regardless of outcome.

- **H1 — surprise beats level.** Cross-sectional ranks describe how a stock's news
  compares with its peers but cannot express how it compares with the stock's own
  normal. The literature associates returns with *abnormal* attention and tone.
- **H2 — one day is too short.** Daily returns are dominated by microstructure
  noise; five days gives any genuine effect room to surface.
- **H3 — coverage relates to volatility.** Attention and volatility are linked in
  the literature, and magnitude is far more forecastable than sign.
- **H4 — sentiment is contemporaneous.** If a headline describes a move that has
  already happened, it cannot forecast the next one — which would explain H0–H2.
- **H5 — a model trained on returns beats one trained on tone.** FinBERT optimises
  for sentiment, which H4 shows is contemporaneous. Training directly on the label
  that is actually wanted should do better, if anything is there.

H1 and H2 are tested by `experiments.py` as classifier configurations. H3 and H4 are
direct correlation tests in `relationships.py`, since neither needs a model. H5 is
`text_model.py`, evaluated through the same walk-forward harness.

| Configuration | Rows | Accuracy | Edge vs baseline | Mean IC | IC sessions | t |
|---|---|---|---|---|---|---|
| H0 level / 1-day | 69,539 | 50.5% | −0.4% | +0.0104 | 2,510 | +2.48 |
| H1 surprise / 1-day | 69,347 | 50.3% | −0.6% | −0.0044 | 2,504 | −1.08 |
| H2 level / 5-day | 69,539 | 51.1% | +0.1% | +0.0192 | 502 | +2.05 |
| H1+H2 surprise / 5-day | 69,347 | 50.3% | −0.7% | −0.0049 | 501 | −0.57 |
| aux combined / 1-day | 69,347 | 50.3% | −0.6% | −0.0014 | 2,504 | −0.34 |

**Minimum detectable mean IC** at each configuration's sample size:

```
H0  level / 1-day        0.0107   (2,510 independent sessions)
H1  surprise / 1-day     0.0104   (2,504 independent sessions)
H2  level / 5-day        0.0241   (  502 independent sessions)
```

H2's floor is more than twice the others because the five-day IC series is strided
by five sessions: consecutive five-day returns share four of their five days, so an
uncorrected t-statistic would substantially overstate significance.

The threshold is derived, not rounded. At five tests the exact two-sided Bonferroni
value is 2.576, and an earlier version used a rounded 2.5 - which was the only
reason a t of 2.54 once printed as significant.

Testing stopped at five classifier configurations. Continuing until something
cleared the threshold would have produced a number, not a finding. H3 to H5 below
ask different questions, each stated in advance and each reported with the control
that bounds it.

### H3 and H4, with controls

```
H4  sentiment -> the return that already happened      IC +0.0238   t  +5.47
    sentiment -> next session's excess return          IC +0.0057   t  +1.33

H3  coverage level -> |next return| (uncontrolled)     IC +0.0483   t +11.63
    trailing volatility -> |next return|               IC +0.2258   t +48.34
    coverage -> volatility residual (controlled)       IC +0.0232   t  +5.66
    within-ticker coverage -> within-ticker |return|   rho +0.0167  t  +4.50
```

H3's uncontrolled figure overstates the effect roughly twofold. Volatility clusters,
so anything correlated with a stock being volatile appears to forecast volatility;
trailing realized vol alone reaches t = +48.34. Coverage survives that control at
t = +5.66, and survives the within-ticker control at t = +4.50.

That second control is the strict one, and passing it is what makes H3 a finding
rather than a restatement. It discards every between-stock difference, so it cannot
be satisfied by "widely covered names are volatile names": a *given stock's* busier
news day predicts *that stock's* next-session move.

It is worth recording that this control read t = −0.14 on 615 sessions, +3.62 on
1,887 and +4.50 on 2,775. The earlier null was a power problem, not an absence.

### H5, a model trained on this project's own data

FinBERT optimises for tone. H4 shows tone is contemporaneous, so `text_model.py`
skips the intermediate step: TF-IDF over headline text into a linear classifier,
trained directly on whether the name beat the cross-section next session, evaluated
through the same walk-forward harness.

A linear model rather than a fine-tuned transformer, deliberately. At 307k short
headlines carrying a signal four prior tests could not detect, 66M parameters would
memorise the training set — and the token weights below are inspectable in a way
transformer weights are not.

```
FinBERT sentiment (H0)                  IC +0.0104   t +2.48
TF-IDF on returns (H5), raw             IC +0.0091   t +2.12
TF-IDF on returns (H5), controlled      IC +0.0004   t +0.11

share of score variance fixed per ticker:  8.1%
```

Trained on 298,690 headlines, it matches FinBERT and no more. The control is
decisive: demeaning each ticker's scores collapses the raw t of +2.12 to **+0.11**.
Headlines name the company they describe, so a bag-of-words model learns *Tesla*
and scores every Tesla headline alike — the heaviest weights are company names and
date fragments, not language about the business.

**This is the third distinct form the same failure took.** Raw volume let the
classifier identify tickers; coverage predicted volatility between stocks but not
within them; company names do it again through text. In a cross-section, any
feature varying more between names than within them will look predictive of
anything else that also varies between names.

---

## 🏗️ Architecture

```
scraper.py             RSS feeds ─────────▶ raw_news, news_tickers
backfill_news.py       FNSPID (streamed) ─▶ raw_news, news_tickers
sentiment_analyzer.py  unscored articles ─▶ sentiment_scores
build_features.py      yfinance ──────────▶ prices;  joined ─▶ daily_features
walkforward.py         daily_features ────▶ out-of-sample predictions
experiments.py         H0-H2: can a classifier predict direction?
relationships.py       H3-H4: what does the same data relate to?
text_model.py          H5: TF-IDF model trained on returns, not tone
backtest.py            predictions ───────▶ equity curve, cost analysis
app.py                 Streamlit dashboard
```

Every stage reads from and writes to SQLite, so stages are independent and
individually re-runnable. A stage that fails halfway leaves no partial state.

| Stage | Module | Status |
|---|---|---|
| News ingestion | `scraper.py` | ✅ Scheduled hourly |
| Historical backfill | `backfill_news.py` | ✅ 2013–2024, 213k headlines |
| Persistence layer | `database.py`, `schema.sql` | ✅ |
| Sentiment scoring | `sentiment_analyzer.py` | ✅ 250.9k headlines scored |
| Feature construction | `build_features.py` | ✅ 73.3k ticker-days |
| Model & evaluation | `alpha_engine.py`, `walkforward.py` | ✅ |
| Direction tests (H0-H2) | `experiments.py` | ✅ 5 configurations |
| Relationship tests (H3-H4) | `relationships.py` | ✅ with controls |
| Own model (H5) | `text_model.py` | ✅ TF-IDF on returns |
| Backtest | `backtest.py` | ✅ |
| Dashboard | `app.py` | ✅ |

**Current data:** 251,237 articles · 250,853 scored · 214,404 price bars
(2011–2026) · 57 tickers · 2,884 sessions · 73,267 feature rows.

---

## 🧠 Design notes

Five decisions account for most of the engineering, and each prevents a specific
way of arriving at a wrong answer.

### Point-in-time correctness

A headline published after the 16:00 ET close cannot inform a position entered at
that close. Attributing it to the session that just ended trains the model on
information that did not exist when the trade would have been placed — a lookahead
bias that inflates results while remaining invisible in every accuracy metric.

Each article stores both the raw UTC publication timestamp and a derived
`session_date`; news after the close, or outside the trading week, rolls to the
next session. When this was first applied to live data it reclassified **half** the
intraday feed output.

Because the raw timestamp is preserved rather than overwritten, adopting a
different execution assumption is a feature rebuild rather than a data migration.

### Articles are deduplicated, not ticker-headline pairs

The same article is frequently syndicated across several ticker feeds. Articles are
keyed by a SHA-256 hash of their canonicalized URL — query strings and tracking
parameters stripped — with a junction table mapping them to tickers, so each
headline is scored exactly once regardless of how many feeds carried it.

### Features are cross-sectionally normalized

Raw volume spans **799×** across this universe and headline counts **43×**. Raw
values therefore identify the company rather than describe the day, and an early
version of the model keyed on exactly that. Features are within-session percentile
ranks.

`sentiment_surprise` and `attention_surprise` measure each observation against the
ticker's own trailing baseline, computed from strictly prior observations —
`shift(1)` before the rolling window, without which each row would be compared
against a baseline containing itself.

### The target is market-relative

**55% of the variance** in a single stock's daily return is the market moving,
which company-level news sentiment cannot forecast. `excess_return` subtracts the
session's cross-sectional mean, leaving the component a stock-selection signal
could plausibly explain.

### Evaluation is walk-forward, not a single split

The corpus ends June 2020, so any single chronological 80/20 split places the COVID
crash in the test set — permanently, regardless of features. Walk-forward refits
monthly on an expanding window and scores each subsequent out-of-sample window,
giving 18 windows across calm markets, the Q4 2018 selloff, the crash and the
recovery. The worst windows turned out to be **August and September 2019**, both
unremarkable months, which ruled out a regime-specific explanation.

---

## 🗄️ Database schema

| Table | Contents |
|---|---|
| `raw_news` | Unique articles, deduplicated by canonical-URL hash |
| `news_tickers` | Junction table associating articles with ticker feeds |
| `sentiment_scores` | Classifier output, keyed by article **and model** |
| `prices` | Daily OHLCV bars |
| `daily_features` | Model-ready matrix, rebuilt on each feature build |
| `v_scored_headlines` | View flattening the news tables for presentation |

The first four are source of truth and are only appended to or upserted; RSS feeds
expose a short rolling window, so a record lost there cannot be recovered.
`daily_features` is derived and regenerated in full, so rows computed under a
superseded feature definition cannot survive alongside current ones.

Every write is an idempotent upsert keyed on a natural key, which is what allows
hourly collection against a feed that repeats the same items for hours.

`sentiment_scores.net_sentiment` is a **generated column** mapping label and
confidence to a signed magnitude, defined once in the schema so the model, the
backtest and the dashboard cannot disagree. All tables are `STRICT`.

---

## 🚀 Getting started

### Prerequisites

* Python 3.10+
* SQLite 3.37 or newer, required for `STRICT` tables. Python bundles its own
  build; check with:

  ```
  python -c "import sqlite3; print(sqlite3.sqlite_version)"
  ```

### Installation

```
git clone https://github.com/ElliottPurdue/nlp-alpha-engine.git
cd nlp-alpha-engine

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Running the pipeline

```
python database.py             # create the database (idempotent)
python scraper.py              # collect current headlines
python backfill_news.py nasdaq --start 2013-01-01   # stream history from FNSPID
python sentiment_analyzer.py   # score everything unscored
python build_features.py       # ingest prices, rebuild the feature matrix
python walkforward.py          # walk-forward evaluation
python experiments.py          # H0-H2: direction, pre-registered
python relationships.py        # H3-H4: volatility and contemporaneity
python text_model.py           # H5: own model, trained on returns
python backtest.py             # equity curve and cost analysis
python inspect_db.py           # summary of database contents
```

```
python -m streamlit run app.py
```

Scoring is decoupled from collection by an anti-join, so a backlog of any size can
be cleared in a single pass whenever convenient, and an interrupted pass resumes
exactly where it stopped.

---

## ⏱️ Automation

`run_pipeline.bat` is the scheduled-task entry point, appending output to
`pipeline.log`:

```
schtasks /Create /TN "NLP Alpha Scraper" /TR <full-path>\run_pipeline.bat /SC HOURLY /F
```

Hourly rather than daily, because each feed exposes only its most recent ~20 items
per ticker and that window turns over within a trading day; measured collection
during market hours runs 19–28 new articles an hour.

The scraper exits non-zero when a run collects nothing, distinguishing unreachable
feeds from feeds that respond but parse to zero items. A scheduler observes only
the exit status, so without this a changed feed format would be reported as success
indefinitely while collecting no data.

---

## ⚠️ Limitations

Stated plainly, because each one bounds the conclusion above.

1. **96.8% of FNSPID records carry no intraday timestamp.** Their position relative
   to the close is unknowable, so they are attributed to the *following* session —
   the only assumption that cannot leak. It is deliberately conservative: a genuine
   same-session effect is shifted a day later and therefore understated.
2. **Universe coverage is uneven.** 54 of 57 tickers appear in the study. The
   nasdaq backfill substantially repaired an earlier sector skew — industrials went
   from 1 of 6 covered to 4 of 6 — but coverage still varies by name, and the
   cross-section width ranges from 29 to 53 names across the sample.
3. **Headlines only, no article bodies.** Sentiment is inferred from titles.
4. **FinBERT is used off the shelf**, not fine-tuned on this corpus.
5. **A two-year gap** separates the backfill (ends January 2024) from live
   collection (began July 2026). The study is fitted on the historical block; the
   live period is reserved as an untouched forward test and is currently too short
   to serve as one.

6. **News density rises sixfold across the sample**, from 1.05 articles per
   ticker-day in 2013 to 6.19 in 2023. The rank features are within-session
   percentiles and the surprise features use each ticker's own trailing baseline, so
   both are insensitive to that trend — but an IC computed across 29 names is
   noisier than one across 53.
7. **Two model classes.** XGBoost with fixed hyperparameters, plus a TF-IDF linear
   model; no ensembles, no fine-tuned transformer.
8. **Costs are a flat 5bps per side.** No market impact, no borrow cost on shorts,
   no slippage modelling. The 42% annual drag is a consequence of daily rebalancing
   specifically; a weekly book would cost roughly a fifth as much, which would not
   be enough to lift a gross Sharpe of 0.72 above a passive 0.88.

---

## 📚 Data attribution

Historical headlines are from **FNSPID** (Financial News and Stock Price
Integration Dataset), licensed **CC BY-NC-4.0** — non-commercial use with
attribution. Two exports are used: `All_external.csv` (2009–2020, headlines only)
and `nasdaq_exteral_data.csv` (2007–2024, with article bodies), both streamed and
filtered rather than downloaded whole.

> Dong, Z., Fan, X., & Peng, Z. (2024). *FNSPID: A Comprehensive Financial News
> Dataset in Time Series.* [arXiv:2402.06698](https://arxiv.org/abs/2402.06698) ·
> [Dataset](https://huggingface.co/datasets/Zihan1004/FNSPID)

Live headlines are collected from Yahoo Finance RSS feeds. Market data is from
Yahoo Finance via `yfinance`. Sentiment scoring uses
[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).

---

## 📊 Tech stack

* **Language:** Python 3.13
* **NLP & ML:** HuggingFace Transformers, PyTorch, XGBoost, scikit-learn
* **Data engineering:** SQLite (WAL mode, STRICT tables, generated columns),
  pandas, NumPy, BeautifulSoup4, lxml
* **Market data:** yfinance
* **Reporting:** Streamlit, Altair, matplotlib
* **Scheduling:** Windows Task Scheduler
