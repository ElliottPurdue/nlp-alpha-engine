"""Tests for the NLP alpha engine.

Written against unittest rather than pytest so the suite runs from a clean clone
with no dependency beyond what the pipeline itself needs.

Coverage is deliberately uneven. It concentrates on the handful of places where a
mistake would not raise, would not look wrong, and would quietly invalidate every
number the project reports: trading-session attribution, the causality of the
surprise baselines, the article dedup key, and the walk-forward train/test split.
Printing and reporting code is left untested.

Run with:
    python -m unittest discover -s tests -t .
"""
