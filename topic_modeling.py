import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

'''
We need topics for each review which summarise them. The function below takes the fitted LDA model,
the feature names (vocabulary), and a count of top words per topic. It returns a dict mapping
topic labels to their top words.
'''
def display_topics(model, feature_names, no_top_words):
    # BUG FIX: The original code had `return` INSIDE the for loop, so it exited after the
    # very first topic and only ever returned a single string. Fixed by building a dict of
    # all topics and returning it after the loop completes.
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        # Filter out common uninformative words for cleaner topic labels
        topic_words = [
            feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]
            if feature_names[i] not in ('book', 'books', 'great', 'nice', 'read', 'good')
        ]
        topics[f'Topic {topic_idx + 1}'] = ' '.join(topic_words)
    return topics

'''
Main function for performing topic modelling on a reviews DataFrame.
Returns a string summary of all topics (for use in analysis_summary.py / app.py display).
'''
def main_topics(reviews_df):
    # BUG FIX: The original used max_df=1 (absolute count of 1 document) and min_df=1, which
    # is far too restrictive for any real dataset — it effectively kept only words appearing in
    # exactly one document. Changed to sensible relative thresholds (max_df=0.95, min_df=2).
    # Also added fillna to prevent errors on missing review text.
    reviews_df = reviews_df.copy()
    reviews_df['review_summary'] = reviews_df['review_summary'].fillna('')

    # Need at least 2 documents to fit meaningfully; return gracefully if not
    if len(reviews_df) < 2:
        return "Not enough reviews to generate topics."

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    try:
        doc_term_matrix = vectorizer.fit_transform(reviews_df['review_summary'])
    except ValueError:
        # Vocabulary is empty after filtering — too few or too short reviews
        return "Could not extract topics: vocabulary too sparse."

    # BUG FIX: n_components must not exceed the number of documents.
    n_components = min(5, len(reviews_df))
    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    lda.fit(doc_term_matrix)

    no_top_words = 10
    topics = display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

    # Return a readable summary string suitable for display in the Streamlit app
    summary = ' | '.join([f"{label}: {words}" for label, words in topics.items()])
    return summary


def main():
    reviews_df = pd.read_csv('Data/filtered_reviews/juvenile fiction_df.csv',
                              usecols=['title', 'review_summary'])
    test = reviews_df.head(1000)
    review_main_topics = main_topics(test)
    print(review_main_topics)

if __name__ == "__main__":
    main()
