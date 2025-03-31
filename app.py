import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # ✅ Fix: Import process.extract

# Load dataset
movies = pd.read_csv('Tamil_movies.csv')

# Display sample dataset
st.write("### Sample Movies from Dataset:")
st.write(movies[['MovieName', 'Genre', 'Director', 'Actor']].head())  

# Remove missing values
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)

# Extract movie names
movie_names = movies["MovieName"].dropna().unique().tolist()

# Function for autocomplete suggestions
def get_suggestions(query, choices, limit=5):
    suggestions = process.extract(query, choices, limit=limit)
    return [match[0] for match in suggestions]

# Streamlit UI
st.title("🎬 Tamil Movie Recommendation System")

# User input with autocomplete
movie_query = st.text_input("Enter a movie name:")

selected_movie = None  # Initialize selected_movie
if movie_query:
    suggestions = get_suggestions(movie_query, movie_names)
    selected_movie = st.selectbox("Did you mean:", suggestions)

# Button to get recommendations
if selected_movie and st.button("Recommend"):
    st.write(f"✅ You selected: **{selected_movie}**")

    # Create content feature
    movies['content'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + movies['Actor']

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation function
    def recommend_movies(title, num_recommendations=5):
        title = title.strip().lower()
        movies['clean_name'] = movies['MovieName'].str.strip().str.lower()

        if not movies['clean_name'].eq(title).any():  # ✅ Fix: Avoid lookup errors
            return ["❌ Movie not found! Please check the spelling."]

        idx = movies[movies['clean_name'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        return movies['MovieName'].iloc[sim_indices]

    # Get recommendations
    recommendations = recommend_movies(selected_movie)

    # Display recommendations
    st.write("### 🎥 Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
