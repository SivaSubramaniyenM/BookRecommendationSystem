import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
import nltk

# Download required NLTK data if not already present
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# I've added this function as previously some of the text for genres was not fully processed and wouldn't match
# any of the genres I had listed - this takes the text and removes words like 'and', removes capitals, commas etc.
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# I'm using the large file we have and trying to split it into the following genres - the genres were chosen
# as previously the file had many very niche genres and so I chose those with the large amounts of books within
# that genre in order to be able to explore the data fully
reviews_df = pd.read_csv('Data/books_and_reviews.csv', usecols=['title', 'categories', 'review_summary',
                                                                 'review_score', 'publisher'])

reviews_df['categories'] = reviews_df['categories'].apply(preprocess_text)

category_list = ['antiques collectibles', 'architecture', 'art', 'bible', 'biography autobiography',
                 'body mind spirit', 'business economics', 'comics graphic novels',
                 'computers', 'cooking', 'crafts hobbies', 'design', 'drama', 'education', 'family relationships',
                 'fiction', 'foreign language study', 'games', 'gardening', 'health fitness', 'history',
                 'house home', 'humor', 'juvenile fiction', 'juvenile nonfiction', 'language arts disciplines',
                 'law', 'literary collections', 'literary criticism', 'mathematics', 'medical', 'music', 'nature',
                 'performing arts', 'pets', 'philosophy', 'photography', 'poetry', 'political science', 'psychology',
                 'reference', 'religion', 'science', 'self-help', 'social science', 'sports recreation', 'study aids',
                 'technology engineering', 'transportation', 'travel', 'true crime', 'young adult fiction']

# BUG FIX: Create the output directory if it doesn't exist.
# Previously files were saved to the current working directory instead of Data/filtered_reviews/
output_dir = 'Data/filtered_reviews'
os.makedirs(output_dir, exist_ok=True)

# BUG FIX: Replaced nested loop (which compared every group against every genre, O(n*m) and fragile)
# with a direct filter per genre using boolean masking — much faster and more reliable.
for genre in category_list:
    genre_df = reviews_df[reviews_df['categories'] == genre]
    if not genre_df.empty:
        filename = os.path.join(output_dir, f"{genre}_df.csv")
        genre_df.to_csv(filename, index=False)
        print(f"Saved {filename}")
    else:
        print(f"No data found for genre: '{genre}' — skipping.")
