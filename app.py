import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('Tamil_movies.csv')
st.write("### Sample Movies from Dataset:")
st.write(movies[['MovieName', 'Genre', 'Director', 'Actor']].head())  # Show first few rows
# Remove missing values in key columns
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)


# Create content feature
movies['content'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + movies['Actor']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title, num_recommendations=5):
    title = title.strip().lower()  # Convert input to lowercase & remove spaces
    movies['clean_name'] = movies['MovieName'].str.strip().str.lower()  # Clean dataset movie names

    if title not in movies['clean_name'].values:
        return ["❌ Movie not found! Please check the spelling."]

    idx = movies[movies['clean_name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return movies['MovieName'].iloc[sim_indices]

# Streamlit UI
st.title("🎬 Tamil Movie Recommendation System")
movie_name = st.text_input("Enter a movie name:")
if st.button("Recommend"):
    recommendations = recommend_movies(movie_name)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
