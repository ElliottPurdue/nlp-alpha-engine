"""Print a summary of current database contents.

Reports table sizes, the distribution of headlines across trading sessions, any
articles carried by more than one ticker feed, and the sentiment label mix.
Opened read-only so it is safe to run while the scheduled pipeline is writing.
"""

import database as db

with db.connect(read_only=True) as conn:
    print("=== rows per table ===")
    for table, count in db.table_counts(conn).items():
        print(f"  {table:<18} {count:>6}")

    # The published_at range within each session reveals the close boundary:
    # a session whose earliest article is timestamped on the previous calendar
    # day is the after-close roll applied by database.to_session_date.
    print("\n=== how news maps onto trading sessions ===")
    for r in conn.execute("""
        SELECT session_date,
               COUNT(*)          AS n,
               MIN(published_at) AS earliest,
               MAX(published_at) AS latest
        FROM raw_news
        GROUP BY session_date
        ORDER BY session_date
    """):
        print(f"  {r['session_date']}  n={r['n']:<4} published {r['earliest']} .. {r['latest']}")

    print("\n=== articles syndicated to more than one ticker ===")
    for r in conn.execute("""
        SELECT n.headline, GROUP_CONCAT(t.ticker) AS tickers
        FROM raw_news n
        JOIN news_tickers t USING (article_id)
        GROUP BY n.article_id
        HAVING COUNT(*) > 1
    """):
        print(f"  [{r['tickers']:<11}] {r['headline'][:60]}")

    print("\n=== sentiment mix ===")
    for r in conn.execute("""
        SELECT sentiment_label, COUNT(*) AS n, ROUND(AVG(net_sentiment), 3) AS avg_net
        FROM sentiment_scores
        GROUP BY sentiment_label
        ORDER BY n DESC
    """):
        print(f"  {r['sentiment_label']:<9} n={r['n']:<4} avg net_sentiment={r['avg_net']}")
