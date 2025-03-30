import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('Tamil_movies.csv')

# Create content feature
movies['content'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + movies['Actor']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title, num_recommendations=5):
    if title not in movies['MovieName'].values:
        return ["Movie not found!"]
    idx = movies[movies['MovieName'] == title].index[0]
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
