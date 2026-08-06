import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_yahoo_news(ticker):
    """Fetches and parses the RSS news feed for a given ticker."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    
    # Mimic a standard web browser to bypass basic anti-bot protections
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Send the HTTP request with the headers
    response = requests.get(url, headers=headers)
    
    # Parse the XML response
    soup = BeautifulSoup(response.content, features="xml")
    
    # Use the updated find_all syntax
    articles = soup.find_all('item')
    
    news_data = []
    for article in articles:
        # Some RSS items might be missing tags, so we use .text safely
        title = article.title.text if article.title else "No Title"
        pub_date = article.pubDate.text if article.pubDate else "No Date"
        link = article.link.text if article.link else "No Link"
        
        news_data.append({
            'ticker': ticker,
            'headline': title,
            'published_at': pub_date,
            'link': link
        })
        
    return news_data

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'JPM']
    all_news = []

    print("Fetching news headlines...")
    for t in tickers:
        print(f"Scraping {t}...")
        all_news.extend(fetch_yahoo_news(t))

    df = pd.DataFrame(all_news)
    
    filename = 'raw_headlines.csv'
    df.to_csv(filename, index=False)
    
    print(f"\nSuccess! Saved {len(df)} headlines to {filename}")
    print(df.head())