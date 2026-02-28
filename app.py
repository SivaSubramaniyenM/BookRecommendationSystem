import streamlit as st
from analysis_summary import recommend_book


def main():
    st.title('Book Recommendation System')
    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the Book Recommendation System")
        st.write("""
            Discover books tailored to your interests! Use the 'Recommend' section to select a genre
            and enter keywords to get personalised book suggestions.

            **How to Use This App:**

            1. Navigate to the **Recommend** tab from the sidebar.
            2. Select your preferred genre from the dropdown menu.
            3. Enter a keyword that interests you (e.g., 'adventure', 'romance').
            4. Click the **Recommend** button to see a list of books tailored to your preferences.

            Explore different genres and discover your next favourite read! Enjoy!
        """)

    elif choice == "Recommend":
        st.subheader("Get Your Book Recommendations")

        genre_select = st.selectbox(
            'Please select your preferred genre:',
            ['antiques collectibles', 'architecture', 'art', 'bible', 'biography autobiography',
             'body mind spirit', 'business economics', 'comics graphic novels', 'computers',
             'cooking', 'crafts hobbies', 'design', 'drama', 'education', 'family relationships',
             'fiction', 'foreign language study', 'games', 'gardening', 'health fitness',
             'history', 'house home', 'humor', 'juvenile fiction', 'juvenile nonfiction',
             'language arts disciplines', 'law', 'literary collections', 'literary criticism',
             'mathematics', 'medical', 'music', 'nature', 'performing arts', 'pets', 'philosophy',
             'photography', 'poetry', 'political science', 'psychology', 'reference', 'religion',
             'science', 'self-help', 'social science', 'sports recreation', 'study aids',
             'technology engineering', 'transportation', 'travel', 'true crime',
             'young adult fiction']
        )

        key_term = st.text_input("Enter a keyword (e.g., 'adventure', 'romance', 'history')")

        if st.button("Recommend"):
            if genre_select and key_term.strip():
                st.write(f"Recommendations for genre: **{genre_select}** | Keyword: **{key_term}**")

                with st.spinner("Finding the best books for you..."):
                    results, topics = recommend_book(genre_select, key_term)

                # BUG FIX: The original code had a broken conditional:
                #   if results.empty → warning (correct)
                #   elif results:   → this raises "ValueError: The truth value of a DataFrame is ambiguous"
                #                     for any non-empty DataFrame, crashing the app.
                # Fixed by using a single `if/else` on results.empty.
                if results.empty:
                    st.warning("No recommendations found. Try a different keyword or genre.")
                else:
                    st.success(f"Found {len(results)} recommendation(s)!")

                    # Display recommended books table with selected columns
                    display_cols = ['title', 'review_score', 'sentiment', 'total_score']
                    # Only show columns that exist (sentiment/total_score are computed)
                    available_cols = [c for c in display_cols if c in results.columns]
                    st.dataframe(results[available_cols].reset_index(drop=True))

                    # Show the topic summary if available
                    if topics and isinstance(topics, str) and topics.strip():
                        st.subheader("Key Themes in These Reviews")
                        st.write(topics)
            else:
                st.error("Please select a genre and enter at least one keyword.")

    else:
        st.subheader("About")
        st.write("""
            This book recommendation system was developed as part of the CFG Degree Summer 2024 group project.
            Our aim is to provide users with personalised book suggestions based on their interests and preferences.
            We hope you find your next great read!

            **Project Team Members:**
            - Eva Morris
            - Wing Hang
            - Srivatsala K A
            - Swarna Dharshini S
        """)


if __name__ == '__main__':
    main()
