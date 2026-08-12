# Deploying collection to a server

How to move news collection off a laptop and onto an always-on host, so
collection gaps stop depending on whether a machine is awake and plugged in.

This is a documented option, not an active deployment. The research in this
repository was completed with collection running locally on a schedule. Keeping
the plan here because the capacity analysis below is the reason the pipeline is
split the way it is.

---

## What runs where, and why

| | Collection host | Research machine |
|---|---|---|
| Scrapes RSS hourly | ✅ | — |
| Stores raw articles | ✅ | ✅ |
| FinBERT scoring | — | ✅ |
| Features, models, backtest | — | ✅ |
| Dashboard | — | ✅ |

The split is forced by the target host's capacity, and it happens to be the right
architecture anyway.

```
1 CPU core, shared with 6 live trading processes
1.0 GB RAM available
3.3 GB disk free
```

`torch` + `transformers` + FinBERT weights is roughly **3 GB installed** against
3.3 GB free, before the database. Worse, inference is CPU-bound: scoring a
backlog would occupy the only core for hours, on a machine executing trades.

The scraping dependencies total about **20 MB** and a few seconds of CPU per hour.

The pipeline already supports this split. Collection and scoring are decoupled by
an anti-join, so scored articles are simply those with a row in
`sentiment_scores`, and a backlog of any size can be cleared later on whatever
machine has the hardware for it.

## What the setup script touches

Everything lives under `~/nlp-alpha-engine`:

```
venv/                 virtual environment, 3 packages
alpha_engine.db       starts empty
run_scrape.sh         nice + ionice + timeout + flock wrapper
rotate_log.sh         caps scrape.log at 5 MB
scrape.log            output
```

Plus **one** crontab line, appended to the two already present:

```
23 * * * * ~/nlp-alpha-engine/rotate_log.sh; ~/nlp-alpha-engine/run_scrape.sh >> ~/nlp-alpha-engine/scrape.log 2>&1
```

It installs **no system packages** and modifies nothing outside that directory.

The scrape runs at `nice -n 19` and `ionice -c 3`, meaning it yields CPU and disk
to every other process on the box. `flock` prevents overlapping runs, and
`timeout 600` stops a hung feed leaving a process resident until the next hour.
Minute 23 rather than 0 keeps it off the top of the hour when scheduled trading
work tends to cluster.

## The database is not copied up

The server starts with an empty database. Your research machine already holds the
history, and its upsert deduplicates on canonical-URL hash, so re-importing an
article it already has only refreshes `last_seen_at`.

That removes the riskiest step: no 181 MB transfer, no possibility of overwriting
a scored database with an unscored one.

At roughly 1,000 articles a day the server's copy grows about 110 MB a year.
Prune it whenever convenient — the research machine is the archive.

---

## Steps

**1. Copy the files across.** Five files, no directories beyond `deploy/`:

```
scp -i <key> database.py schema.sql scraper.py export_articles.py inspect_db.py \
    ubuntu@<host>:~/nlp-alpha-engine/
scp -i <key> deploy/requirements-server.txt deploy/setup_server.sh \
    ubuntu@<host>:~/nlp-alpha-engine/deploy/
```

Create the directories first:

```
ssh -i <key> ubuntu@<host> 'mkdir -p ~/nlp-alpha-engine/deploy'
```

**2. Run the setup.**

```
ssh -i <key> ubuntu@<host> 'chmod +x ~/nlp-alpha-engine/deploy/setup_server.sh && ~/nlp-alpha-engine/deploy/setup_server.sh'
```

It ends with one immediate scrape so a failure surfaces straight away instead of
an hour later.

**3. Verify.**

```
ssh -i <key> ubuntu@<host> 'tail -20 ~/nlp-alpha-engine/scrape.log; crontab -l'
```

## Syncing articles back

Run on the research machine whenever you want to score:

```
ssh -i <key> ubuntu@<host> 'cd nlp-alpha-engine && venv/bin/python export_articles.py' | python import_articles.py
python sentiment_analyzer.py
```

The export sends a rolling 7-day window rather than tracking a watermark.
Re-importing costs nothing because the upsert is idempotent, and a window is
self-healing: a sync missed for three days is repaired by the next one, where
watermark bookkeeping would leave a permanent hole and not tell you.

Adjust with `--days N` after an unusually long gap.

## Turning the laptop's own scraper off

Once collection is running on the server, the local scheduled task is redundant
and only produces articles the server would collect anyway:

```
schtasks /Delete /TN "NLP Alpha Scraper" /F
```

Worth keeping it until you have confirmed a few clean hourly runs on the server.

## Undoing all of it

```
ssh -i <key> ubuntu@<host> 'crontab -l | grep -v nlp-alpha-engine | crontab -; rm -rf ~/nlp-alpha-engine'
```

That is the complete footprint. No system packages, no services, no users, no
firewall rules.

## Why the dashboard is not here

That host holds credentials for live trading. A public web service on it widens
the attack surface materially, and the `fail2ban` already running suggests that
matters to you.

Two alternatives keep it out of the blast radius: bind Streamlit to `127.0.0.1`
and reach it through an SSH tunnel when wanted, or deploy publicly to Streamlit
Community Cloud from a 27.8 MB database snapshot.
