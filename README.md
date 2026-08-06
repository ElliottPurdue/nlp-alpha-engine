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
    git clone https://github.com/ElliottPurdue/nlp-alpha-engine.git
    cd nlp-alpha-engine

2. Create and activate a virtual environment:
    python -m venv venv
    venv\Scripts\activate

3. Install required packages:
    pip install requests beautifulsoup4 pandas lxml torch transformers yfinance xgboost scikit-learn

### Running the Pipeline

Step 1: Fetch raw financial headlines
    python scraper.py

Step 2: Run FinBERT sentiment analysis
    python sentiment_analyzer.py

Step 3: Merge market data & train XGBoost model
    python alpha_engine.py

---

## 📊 Tech Stack
* **Language:** Python
* **NLP & ML:** HuggingFace Transformers, PyTorch, XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy, BeautifulSoup4, YFinance
