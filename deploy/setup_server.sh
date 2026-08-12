#!/usr/bin/env bash
#
# Sets up hourly news collection on the server. Run ON the server, after the
# files listed in deploy/README.md have been copied across.
#
# Everything it creates lives under $TARGET. It installs no system packages,
# touches nothing outside that directory, and adds exactly one crontab line.
# deploy/README.md documents how to undo all of it.
#
# Safe to re-run: each step checks before acting.

set -euo pipefail

TARGET="${TARGET:-$HOME/nlp-alpha-engine}"
CRON_MINUTE="${CRON_MINUTE:-23}"

echo "=> target directory: $TARGET"
cd "$TARGET"

for required in database.py schema.sql scraper.py export_articles.py \
                deploy/requirements-server.txt; do
    if [ ! -f "$required" ]; then
        echo "   MISSING: $required -- copy the files listed in deploy/README.md first" >&2
        exit 1
    fi
done
echo "   all required files present"

# --------------------------------------------------------------------------
echo "=> virtual environment"
if [ -d venv ]; then
    echo "   already exists, reusing"
else
    python3 -m venv venv
    echo "   created"
fi
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r deploy/requirements-server.txt
echo "   dependencies installed ($(./venv/bin/pip list 2>/dev/null | wc -l) packages)"

# --------------------------------------------------------------------------
echo "=> database"
# Starts empty on purpose. The research machine already holds the history, and
# its upsert deduplicates on canonical-URL hash at import, so nothing needs to be
# copied up and there is no large transfer to get wrong.
./venv/bin/python database.py

# --------------------------------------------------------------------------
echo "=> collection wrapper"
cat > run_scrape.sh <<'WRAPPER'
#!/usr/bin/env bash
# Hourly collection. Kept deliberately modest: this host runs live trading, and
# a news scrape must never be the reason a strategy misses a tick.
#
#   nice/ionice   yields CPU and disk to everything else on the box
#   timeout       a hung feed cannot leave a process resident until the next run
#   flock         overlapping runs are impossible if one is slow
cd "$(dirname "$0")"
exec flock -n .scrape.lock \
     nice -n 19 ionice -c 3 \
     timeout 600 ./venv/bin/python scraper.py
WRAPPER
chmod +x run_scrape.sh
echo "   wrote run_scrape.sh"

# --------------------------------------------------------------------------
echo "=> log rotation"
cat > rotate_log.sh <<'ROTATE'
#!/usr/bin/env bash
# Keeps scrape.log from growing without bound on a 6.8 GB disk.
cd "$(dirname "$0")"
if [ -f scrape.log ] && [ "$(stat -c%s scrape.log)" -gt 5242880 ]; then
    mv scrape.log scrape.log.1
fi
ROTATE
chmod +x rotate_log.sh
echo "   wrote rotate_log.sh"

# --------------------------------------------------------------------------
echo "=> crontab"
ENTRY="$CRON_MINUTE * * * * $TARGET/rotate_log.sh; $TARGET/run_scrape.sh >> $TARGET/scrape.log 2>&1"

if crontab -l 2>/dev/null | grep -qF "$TARGET/run_scrape.sh"; then
    echo "   entry already present, leaving it alone"
else
    # Appends to the existing crontab rather than replacing it. There are already
    # two entries on this host and they are not ours to disturb.
    ( crontab -l 2>/dev/null; echo "$ENTRY" ) | crontab -
    echo "   added: $ENTRY"
fi

# --------------------------------------------------------------------------
echo
echo "=> verifying with one immediate run"
./run_scrape.sh && echo "   scrape completed" || echo "   scrape FAILED -- see output above"

echo
echo "done. collection runs at :$CRON_MINUTE past each hour."
echo "  logs:   tail -f $TARGET/scrape.log"
echo "  status: $TARGET/venv/bin/python $TARGET/inspect_db.py  (if copied)"
