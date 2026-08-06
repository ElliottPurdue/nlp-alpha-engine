import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

# Suppress yfinance warnings for cleaner terminal output
warnings.filterwarnings('ignore')

def calculate_net_sentiment(row):
    """Converts FinBERT string labels into numerical values."""
    if row['sentiment_label'] == 'positive':
        return row['sentiment_score']
    elif row['sentiment_label'] == 'negative':
        return -row['sentiment_score']
    else:
        return 0.0

def build_alpha_engine():
    print("Loading and aggregating sentiment data...")
    df_sent = pd.read_csv('scored_headlines.csv')
    
    # Standardize the dates
    df_sent['published_at'] = pd.to_datetime(df_sent['published_at'], utc=True)
    df_sent['date'] = df_sent['published_at'].dt.date
    
    # Apply numerical conversion
    df_sent['net_sentiment'] = df_sent.apply(calculate_net_sentiment, axis=1)
    
    # Aggregate daily sentiment per ticker
    daily_sentiment = df_sent.groupby(['ticker', 'date']).agg(
        mean_sentiment=('net_sentiment', 'mean'),
        sum_sentiment=('net_sentiment', 'sum'),
        headline_count=('headline', 'count')
    ).reset_index()

    # Convert date back to string for easier merging with yfinance data
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

    tickers = daily_sentiment['ticker'].unique()
    all_data = []

    print("Fetching historical market data from Yahoo Finance...")
    for ticker in tickers:
        # Fetch 1 month of historical data to ensure we overlap with our news dates
        stock = yf.Ticker(ticker)
        prices = stock.history(period="1mo")
        prices.reset_index(inplace=True)
        
        # Ensure timezone-naive dates for merging
        prices['Date'] = pd.to_datetime(prices['Date']).dt.tz_localize(None)
        
        # Calculate next day's return (This is our predictive target)
        prices['next_day_return'] = prices['Close'].shift(-1) / prices['Close'] - 1
        prices['target'] = np.where(prices['next_day_return'] > 0, 1, 0)
        
        # Add ticker column for merging
        prices['ticker'] = ticker
        
        # Keep only necessary price features
        prices = prices[['Date', 'ticker', 'Close', 'Volume', 'target']]
        prices.rename(columns={'Date': 'date'}, inplace=True)
        
        all_data.append(prices)

    df_prices = pd.concat(all_data)
    
    print("Merging sentiment features with market data...")
    # Merge on ticker and date
    merged_df = pd.merge(daily_sentiment, df_prices, on=['ticker', 'date'], how='inner')
    
    # Drop rows where target is NaN (the very last day of price data won't have a "next day" return yet)
    merged_df.dropna(subset=['target'], inplace=True)
    
    if len(merged_df) < 5:
        print("\nNotice: Not enough historical days of overlap to effectively train the model.")
        print(f"Total overlapping days found: {len(merged_df)}. Let the scraper run for a few more days!")
        print("Here is the merged feature set prepared for the ML model:")
        print(merged_df.head())
        return

    print(f"Training XGBoost Model on {len(merged_df)} data points...")
    
    # Define features (X) and target (y)
    features = ['mean_sentiment', 'sum_sentiment', 'headline_count', 'Volume']
    X = merged_df[features]
    y = merged_df['target']
    
    # Split the data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the XGBoost Classifier
    model = XGBClassifier(eval_metric='logloss', max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    build_alpha_engine()