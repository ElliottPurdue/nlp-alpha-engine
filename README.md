# NLP-Driven Financial Sentiment & Alpha Engine

A Python data pipeline that collects financial news headlines, scores them with a
finance-specific transformer model (**FinBERT**), aligns the resulting sentiment
signals to tradable market sessions, and tests their predictive value with an
**XGBoost** classifier.

The emphasis is as much on the data engineering as the modelling: a sentiment
signal is only worth backtesting if the data behind it is deduplicated, ingested
idempotently, and correct as of the moment a trade could have been placed.

---

## 📌 Project Status

Storage has been migrated from flat CSV files to SQLite. The pipeline is
partially migrated:

| Stage | Module | Status |
|---|---|---|
| News ingestion | `scraper.py` | ✅ SQLite-backed, scheduled hourly |
| Persistence layer | `database.py`, `schema.sql` | ✅ Complete |
| Sentiment scoring | `sentiment_analyzer.py` | ⚠️ Still reads the legacy CSV extracts |
| Feature construction | `build_features.py` | 🔜 Not yet written |
| Model training | `alpha_engine.py` | ⚠️ Still reads the legacy CSV extracts |
| Backtesting | — | 🔜 Planned (`vectorbt`) |
| Dashboard | `app.py` | 🔜 Planned (`streamlit`) |

**No performance results are reported yet.** The database holds only a few days
of headlines, far short of what a meaningful backtest requires. History
accumulates hourly and cannot be backfilled, because the source feed exposes only
its most recent items per ticker.

---

## 🏗️ Architecture

```
scraper.py            RSS feeds ──▶ raw_news, news_tickers
sentiment_analyzer.py unscored headlines ──▶ sentiment_scores
build_features.py     yfinance ──▶ prices;  joined ──▶ daily_features
alpha_engine.py       daily_features ──▶ XGBoost classifier
```

Each stage reads from and writes to SQLite, so stages are independent and
re-runnable. A stage that fails halfway leaves no partial state behind.

---

## 🧠 Design Notes

Three decisions account for most of the engineering here.

### Point-in-time correctness

A headline published after the 16:00 ET close cannot inform a position entered at
that close. Attributing it to the session that just ended would train the model on
information that did not exist when the trade would have been placed — a lookahead
bias that inflates backtest performance while remaining invisible in every
accuracy metric.

Each article therefore stores both the raw UTC publication timestamp and a derived
`session_date`. Headlines published after the close, or outside the trading week,
roll forward to the next session. In practice this reclassifies a substantial
share of intraday feed output.

Because the raw timestamp is preserved rather than overwritten, adopting a
different execution assumption — next-open entry, say — is a feature rebuild
rather than a data migration.

### Articles are deduplicated, not ticker-headline pairs

The same article is frequently syndicated across several ticker feeds. Storing one
row per (ticker, headline) pair would duplicate the text and run the classifier
repeatedly on identical input. Articles are keyed instead by a SHA-256 hash of
their canonicalized URL — query strings and tracking parameters stripped — with a
junction table mapping them to tickers. Each headline is scored exactly once
regardless of how many feeds carried it.

### Ingestion is idempotent

Every write is an upsert keyed on a natural key, so the pipeline can run at any
frequency — after a failure, twice by the scheduler, or manually during debugging
— without duplicating a row. This is what makes hourly collection viable against a
feed that repeats the same items for hours at a time.

---

## 🗄️ Database Schema

| Table | Contents |
|---|---|
| `raw_news` | Unique articles, deduplicated by canonical-URL hash |
| `news_tickers` | Junction table associating articles with ticker feeds |
| `sentiment_scores` | Classifier output, keyed by article **and model** |
| `prices` | Daily OHLCV bars |
| `daily_features` | Model-ready feature matrix, rebuilt on each feature build |
| `v_scored_headlines` | View flattening the three news tables for presentation |

The first four tables are the source of truth and are only ever appended to or
upserted. `daily_features` is derived and dropped and regenerated in full, so rows
computed under a superseded feature definition cannot survive alongside current
ones.

`sentiment_scores.net_sentiment` is a **generated column** that maps label and
confidence to a signed magnitude. Defining it in the schema rather than in
application code guarantees that the model, the backtest and the dashboard cannot
disagree about what a given headline is worth.

All tables are declared `STRICT`, so a value whose type does not match its column
is rejected at write time instead of surfacing as a failure deep inside model
training.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* SQLite 3.37 or newer, required for `STRICT` tables. Python bundles its own
  build; check it with:

  ```
  python -c "import sqlite3; print(sqlite3.sqlite_version)"
  ```

### Installation

```
git clone https://github.com/ElliottPurdue/nlp-alpha-engine.git
cd nlp-alpha-engine

python -m venv venv
venv\Scripts\activate

pip install requests beautifulsoup4 lxml pandas numpy torch transformers yfinance xgboost scikit-learn
```

### Running the pipeline

Create the database (idempotent — safe to re-run):

```
python database.py
```

Collect headlines:

```
python scraper.py
```

Inspect what has been collected:

```
python inspect_db.py
```

The sentiment and training stages still read the legacy CSV extracts and are
mid-migration:

```
python sentiment_analyzer.py
python alpha_engine.py
```

`migrate_csv.py` imports the pre-database CSV extracts. It is retained rather than
deleted: the database file is not version controlled, so that script and those
extracts together are the recovery path for the earliest headlines, which have
since rotated out of the source feed.

---

## ⏱️ Automation

`run_pipeline.bat` is the entry point for a scheduled task and appends all output
to `pipeline.log`. Registered on Windows with:

```
schtasks /Create /TN "NLP Alpha Scraper" /TR <full-path>\run_pipeline.bat /SC HOURLY /F
```

Hourly rather than daily, because each feed exposes only its most recent items and
that window turns over well within a trading day; a daily run would silently miss
most of the news flow.

The scraper exits non-zero when a run collects nothing at all, distinguishing
unreachable feeds from feeds that respond but parse to zero items. A scheduler
observes only the exit status, so without this a changed feed format would be
reported as a successful run indefinitely while collecting no data.

---

## 🗺️ Roadmap

- [ ] Migrate sentiment scoring and model training onto the database
- [ ] Feature build with holiday-aware trading-calendar alignment
- [ ] Walk-forward backtest with an equity curve against a buy-and-hold baseline (`vectorbt`)
- [ ] Streamlit dashboard: live headline feed and rolling sentiment against price
- [ ] Expand the ticker universe beyond the current five symbols

---

## 📊 Tech Stack

* **Language:** Python 3.13
* **NLP & ML:** HuggingFace Transformers, PyTorch, XGBoost, scikit-learn
* **Data engineering:** SQLite (WAL mode), pandas, NumPy, BeautifulSoup4, lxml
* **Market data:** yfinance
* **Scheduling:** Windows Task Scheduler
