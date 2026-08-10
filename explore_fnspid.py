"""One-off inspection of the FNSPID historical news dataset.

Streams a sample from HuggingFace without downloading the full 5.7 GB file, and
reports schema, timestamp format and ticker coverage, so that the backfill is
written against what the data actually contains rather than an assumption.
"""

import pandas as pd

from scraper import TICKERS

FNSPID_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
    "Stock_news/All_external.csv"
)

SAMPLE_ROWS = 50_000


def main():
    print(f"Streaming first {SAMPLE_ROWS:,} rows (the file itself is 5.7 GB)...")

    # dtype=str disables type inference: the published dataset has mixed types
    # in some columns, and inference on a partial read would guess differently
    # than it will on the full pass.
    frame = pd.read_csv(FNSPID_URL, nrows=SAMPLE_ROWS, dtype=str)

    print(f"\n=== columns ({len(frame.columns)}) ===")
    for col in frame.columns:
        values = frame[col].dropna()
        example = str(values.iloc[0])[:55] if len(values) else "-"
        print(f"  {col:<22} {len(values):>7,} non-null   e.g. {example}")

    date_col = next((c for c in frame.columns if c.lower() == "date"), frame.columns[0])
    print(f"\n=== timestamp column: '{date_col}' ===")
    print("  raw samples:")
    for value in frame[date_col].dropna().head(5):
        print(f"    {value}")

    parsed = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    print(f"  parsed OK:      {parsed.notna().sum():,} of {len(frame):,}")
    print(f"  range:          {parsed.min()}  ..  {parsed.max()}")

    # A timestamp of exactly midnight indicates a date-only record, which cannot
    # be placed relative to the market close.
    midnight = (parsed.dt.hour == 0) & (parsed.dt.minute == 0) & (parsed.dt.second == 0)
    print(f"  date-only rows: {midnight.sum():,} ({midnight.mean():.1%})")
    print(f"  timezone aware: {parsed.dt.tz is not None}")

    symbol_col = next((c for c in frame.columns if "symbol" in c.lower()), None)
    if symbol_col:
        print(f"\n=== ticker coverage: '{symbol_col}' ===")
        symbols = frame[symbol_col].dropna()
        present = sorted(set(symbols) & set(TICKERS))
        print(f"  distinct symbols in sample: {symbols.nunique():,}")
        print(f"  of your 57: {len(present)} present -> {present}")

        # If the file is ordered by symbol, a head sample is not representative
        # of coverage and only the schema findings above can be trusted.
        ordered = symbols.tolist()
        print(f"  file appears sorted by symbol: {ordered == sorted(ordered)}")

    print(f"\n  sample date span suggests file is sorted by date: "
          f"{parsed.dropna().is_monotonic_increasing}")


if __name__ == "__main__":
    main()