"""Long/short equity curve from walk-forward predictions.

Dollar-neutral, rebalanced every session: long the top quintile by score, short the
bottom, equally weighted. That follows from what the model predicts -- relative
performance within a session -- and the return is the spread between the baskets,
so the market's direction drops out.

Written in pandas rather than through a backtesting library. The portfolio is simple
enough that twenty lines of explicit accounting is easier to audit than a library's
conventions about weights, fills and rebalance timing.

Gross and net are both reported. Net matters more than it looks: a daily-rebalanced
book with four names a side turns over almost completely each session, so cost alone
imposes a drag any real signal would have to clear first.
"""

import numpy as np
import pandas as pd

from walkforward import walk_forward

# Fraction taken on each side. A quintile is about four names at the median session
# width of 21.
SIDE_FRACTION = 0.2

# Below this, a basket is a bet on two companies rather than on a signal.
MIN_NAMES_PER_SIDE = 2

# One-way cost in basis points of notional traded, covering commission and spread.
COST_BPS = 5.0

TRADING_DAYS = 252

EQUITY_CURVE_PATH = "equity_curve.png"


def _max_drawdown(equity):
    return (equity / equity.cummax() - 1).min()


def _annualized_stats(returns, label):
    return {
        "label": label,
        "total": (1 + returns).prod() - 1,
        "annualized": (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1,
        "volatility": returns.std() * np.sqrt(TRADING_DAYS),
        "sharpe": (returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
                   if returns.std() else 0.0),
        "max_drawdown": _max_drawdown((1 + returns).cumprod()),
        "hit_rate": (returns > 0).mean(),
    }


def build_portfolio(predictions):
    """Per-session portfolio returns, turnover and benchmark, indexed by date."""
    records = []
    previous_weights = {}

    for session, group in predictions.groupby("session_date"):
        per_side = int(len(group) * SIDE_FRACTION)
        if per_side < MIN_NAMES_PER_SIDE:
            continue

        ordered = group.sort_values("score", ascending=False)
        longs, shorts = ordered.head(per_side), ordered.tail(per_side)

        # Identical whether computed on raw or excess returns, since the session
        # mean cancels between the baskets. Raw is what the book actually earns.
        gross = longs["fwd_return"].mean() - shorts["fwd_return"].mean()

        weights = {t: 1.0 / per_side for t in longs["ticker"]}
        weights.update({t: -1.0 / per_side for t in shorts["ticker"]})

        # Notional traded per unit of capital. A full rotation of both sides reads
        # as 4.0: one unit out and one in, on each side.
        touched = set(weights) | set(previous_weights)
        turnover = sum(abs(weights.get(t, 0.0) - previous_weights.get(t, 0.0))
                       for t in touched)
        previous_weights = weights

        records.append({
            "session_date": session,
            "gross": gross,
            "cost": turnover * COST_BPS / 10_000,
            "turnover": turnover,
            # Equal-weighting every name with news that session: the long-only
            # alternative an investor could have held instead.
            "benchmark": group["fwd_return"].mean(),
        })

    book = pd.DataFrame(records).set_index("session_date")
    book["net"] = book["gross"] - book["cost"]
    return book


def plot_equity_curve(book, path=EQUITY_CURVE_PATH):
    """Write the curve to PNG if matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n  (matplotlib not installed; skipping {path})")
        return False

    figure, axes = plt.subplots(figsize=(11, 5.5))
    for column, label, style in (
        ("gross", "Long/short, gross", "-"),
        ("net", f"Long/short, net of {COST_BPS:.0f}bps", "-"),
        ("benchmark", "Equal-weight universe", "--"),
    ):
        axes.plot((1 + book[column]).cumprod(), style, linewidth=1.4, label=label)

    axes.axhline(1.0, color="grey", linewidth=0.8, alpha=0.6)
    axes.set_title("Cross-sectional news-sentiment strategy, walk-forward "
                   "out-of-sample")
    axes.set_ylabel("Cumulative growth of 1")
    axes.legend(frameon=False)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    print(f"\n  wrote {path}")
    return True


def period_breakdown(book, freq="QE"):
    """Returns by calendar period.

    A headline Sharpe implies steadiness that a concentrated return does not have,
    and this is the cheapest way to see which one you have.
    """
    return book.groupby(pd.Grouper(freq=freq))[["gross", "net", "benchmark"]].sum()


def run():
    print("Generating walk-forward predictions...")
    predictions, _ = walk_forward()

    book = build_portfolio(predictions)
    print(f"  {len(book)} rebalanced sessions "
          f"({book.index.min().date()} .. {book.index.max().date()})")

    rows = [
        _annualized_stats(book["gross"], "long/short gross"),
        _annualized_stats(book["net"], f"long/short net ({COST_BPS:.0f}bps)"),
        _annualized_stats(book["benchmark"], "equal-weight universe"),
    ]

    print(f"\n  {'strategy':<26}{'total':>9}{'ann.':>9}{'vol':>8}"
          f"{'Sharpe':>9}{'maxDD':>9}{'hit':>7}")
    print("  " + "-" * 76)
    for row in rows:
        print(f"  {row['label']:<26}{row['total']:>+9.1%}{row['annualized']:>+9.1%}"
              f"{row['volatility']:>8.1%}{row['sharpe']:>+9.2f}"
              f"{row['max_drawdown']:>+9.1%}{row['hit_rate']:>7.1%}")

    mean_turnover = book["turnover"].mean()
    annual_drag = book["cost"].mean() * TRADING_DAYS
    print(f"\n  mean turnover per session   {mean_turnover:.2f}x capital")
    print(f"  cost drag                   {annual_drag:.1%} per year")

    quarters = period_breakdown(book)
    print(f"\n  {'quarter':<12}{'gross':>10}{'net':>10}{'benchmark':>12}")
    print("  " + "-" * 44)
    for period, row in quarters.iterrows():
        print(f"  {str(period.date()):<12}{row['gross']:>+10.2%}"
              f"{row['net']:>+10.2%}{row['benchmark']:>+12.2%}")

    # This is what reconciles the backtest with a near-zero information
    # coefficient. A return arriving almost entirely in one quarter is an episode,
    # and the IC -- weighting sessions equally instead of by magnitude -- is what
    # catches that. Reported so the headline Sharpe cannot be read alone.
    total_gross = book["gross"].sum()
    best_quarter = quarters["gross"].max()
    negative_quarters = (quarters["gross"] < 0).sum()

    print(f"\n  gross total across all sessions      {total_gross:>+8.2%}")
    print(f"  best single quarter                  {best_quarter:>+8.2%}")
    if total_gross:
        print(f"  gross excluding the best quarter     "
              f"{total_gross - best_quarter:>+8.2%}")
    print(f"  quarters with negative gross return  {negative_quarters} of {len(quarters)}")

    plot_equity_curve(book)
    return book


if __name__ == "__main__":
    run()
