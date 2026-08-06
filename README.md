# NLP-Driven Financial Sentiment & Alpha Engine

A modular Python data pipeline that scrapes financial news headlines, extracts sentiment signals using a fine-tuned transformer model (**FinBERT**), and evaluates market return predictability using an **XGBoost** binary classifier.

---

## 🏗️ Architecture Overview
1. **Data Ingestion (`scraper.py`):** Fetches real-time RSS news feeds for target equity tickers.
2. **NLP Sentiment Pipeline (`sentiment_analyzer.py`):** Passes unstructured text through `ProsusAI/finbert` to compute directional sentiment labels and confidence metrics.
3. **Market Alignment & Model Training (`alpha_engine.py`):** Merges aggregated daily sentiment vectors with historical market price data (`yfinance`) to train an XGBoost classifier predicting 48-hour forward return targets.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+

### Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/ElliottPurdue/nlp-alpha-engine.git](https://github.com/ElliottPurdue/nlp-alpha-engine.git)
   cd nlp-alpha-engine
