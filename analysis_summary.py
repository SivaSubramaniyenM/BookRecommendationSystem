import pandas as pd
from topic_modeling import main_topics
from sentiment_analysis import apply_sentiment_analysis
import os
from fuzzywuzzy import fuzz

'''
Returns True if any keyword (or a sufficiently similar word) is found in the review text.
Uses fuzzy matching so minor spelling differences still count as a match.
'''
def has_keywords(text, keywords):
    if not isinstance(text, str):
        return False
    text = text.lower()
    words = text.split()
    threshold = 80
    for keyword in keywords:
        for word in words:
            if fuzz.ratio(keyword.lower(), word) >= threshold:
                return True
    return False

'''
Takes a dataframe and a list of keywords, and returns a list of dicts for books whose reviews
contain at least one of the given keywords. Each title is only included once (de-duplicated).
'''
def matched_books(reviews_df, keywords):
    matching_books = []
    seen_titles = set()

    for _, review in reviews_df.iterrows():
        keyword_score = 1 if has_keywords(review['review_summary'], keywords) else 0
        if review['title'] not in seen_titles and keyword_score == 1:
            matching_books.append({
                'title': review['title'],
                'categories': review['categories'],
                'review_summary': review['review_summary'],
                'review_score': review['review_score'],
                'publisher': review['publisher'],
                'keyword match': keyword_score
            })
            seen_titles.add(review['title'])

    return matching_books

'''
Main recommendation function used by app.py.
Given a genre and a keyword string, returns:
  - top_10_recommended: DataFrame of up to 10 books ranked by combined sentiment + review score
  - review_topics: string summarising the topics found in those top reviews
Returns (empty DataFrame, []) if no matches are found.
'''
def recommend_book(genre, key_term):
    base_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(base_path, f"Data/filtered_reviews/{genre}_df.csv")

    # BUG FIX: Added a clear FileNotFoundError message so users know what went wrong
    # rather than getting an opaque pandas crash.
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}. Run Genre_filter.py first.")
        return pd.DataFrame(), []

    reviews_df = pd.read_csv(file_path, usecols=['title', 'categories', 'review_summary',
                                                   'review_score', 'publisher'])

    keywords = key_term.strip().split()

    # BUG FIX: keywords list was split but could contain empty strings if key_term had extra spaces.
    keywords = [k for k in keywords if k]

    if not keywords:
        return pd.DataFrame(), []

    books_list = matched_books(reviews_df, keywords)

    if len(books_list) == 0:
        return pd.DataFrame(), []

    # BUG FIX: matched_books returns a plain list of dicts; apply_sentiment_analysis now
    # handles conversion internally, but we convert here explicitly for clarity.
    matching_books = pd.DataFrame(books_list)

    # Apply sentiment analysis
    recommended_books = apply_sentiment_analysis(matching_books)

    # Calculate composite score: average of star rating and VADER compound sentiment
    recommended_books['total_score'] = (
        recommended_books['review_score'] + recommended_books['sentiment']
    ) / 2

    # Sort by composite score descending
    recommended_books = recommended_books.sort_values(by='total_score', ascending=False)

    # Return top 10
    top_10_recommended = recommended_books.head(10)
    review_topics = main_topics(top_10_recommended)

    return top_10_recommended, review_topics


def main():
    keywords = ['serious', 'melancholy']
    genre = 'biography autobiography'

    reviews_df = pd.read_csv(f"Data/filtered_reviews/{genre}_df.csv",
                              usecols=['title', 'categories', 'review_summary',
                                       'review_score', 'publisher'])

    books_list = matched_books(reviews_df, keywords)

    if not books_list:
        print("No matching books found.")
        return

    # BUG FIX: apply_sentiment_analysis expects a DataFrame, not a list
    matching_books = pd.DataFrame(books_list)
    recommended_books = apply_sentiment_analysis(matching_books)

    recommended_books['total_score'] = (
        recommended_books['review_score'] + recommended_books['sentiment']
    ) / 2
    recommended_books = recommended_books.sort_values(by='total_score', ascending=False)

    top_10_recommended = recommended_books.head(10)
    review_topics = main_topics(top_10_recommended)

    print('Recommended books:')
    print(top_10_recommended[['title', 'review_score', 'sentiment', 'total_score']])
    print('Reviews noted these books as:', review_topics)

if __name__ == "__main__":
    main()
