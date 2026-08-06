import pandas as pd
from transformers import pipeline

def analyze_sentiment():
    print("Loading FinBERT model (this may take a minute on the first run)...")
    
    # Initialize the FinBERT sentiment analysis pipeline
    # The 'ProsusAI/finbert' model is specifically trained on financial sentiment
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    print("Loading raw headlines...")
    df = pd.read_csv('raw_headlines.csv')
    
    print("Analyzing sentiment for each headline...")
    
    # Apply the pipeline to the 'headline' column
    # The pipeline returns a list of dictionaries: [{'label': 'positive', 'score': 0.85}, ...]
    results = sentiment_pipeline(df['headline'].tolist())
    
    # Extract labels and confidence scores into new columns
    df['sentiment_label'] = [res['label'] for res in results]
    df['sentiment_score'] = [res['score'] for res in results]
    
    # Save the processed data with the new ML features
    filename = 'scored_headlines.csv'
    df.to_csv(filename, index=False)
    
    print(f"\nSuccess! Saved sentiment scores to {filename}")
    
    # Preview the results, focusing on the new ML columns
    print(df[['ticker', 'sentiment_label', 'sentiment_score', 'headline']].head())

if __name__ == "__main__":
    analyze_sentiment()