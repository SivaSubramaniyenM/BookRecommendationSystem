import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download vader lexicon which analyses the sentiment of the text and creates an instance of an nltk class which uses vader
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

'''
Takes text as input and returns the compound sentiment score: a normalized score which summarizes
sentiment from -1 (most negative) to +1 (most positive).
'''
def analyze_sentiment(review_text):
    # BUG FIX: Guard against non-string input (NaN, float etc.) which caused a TypeError
    # previously when VADER received a non-string value.
    if not isinstance(review_text, str) or review_text.strip() == '':
        return 0.0
    sentiment_score = sia.polarity_scores(review_text)
    return sentiment_score['compound']  # Compound score as overall sentiment measure

def apply_sentiment_analysis(reviews_df):
    # BUG FIX: reviews_df may arrive as a plain list of dicts (from matched_books in analysis_summary.py).
    # Convert to DataFrame first if needed so .apply() works correctly.
    if not isinstance(reviews_df, pd.DataFrame):
        reviews_df = pd.DataFrame(reviews_df)

    # Apply sentiment to the review_summary column, filling any NaN values with empty string
    reviews_df['sentiment'] = reviews_df['review_summary'].fillna('').apply(analyze_sentiment)
    return reviews_df

def main():
    reviews_df = pd.read_csv('Data/filtered_reviews/architecture_df.csv')
    sentiment_df = apply_sentiment_analysis(reviews_df)
    print(sentiment_df[['title', 'sentiment']].head())

if __name__ == "__main__":
    main()
