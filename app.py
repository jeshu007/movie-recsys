import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np

# ✅ Function to add custom styles (without background image)
def add_custom_styles():
    css_code = """
    <style>
    body, .stTextInput, .stSelectbox {
        color: black !important;
        font-weight: bold;
    }
    .movie-title {
        color: red;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        margin-top: 5px;
    }
    div.stButton > button {
        background-color: green !important;
        color: white !important;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        font-size: 18px;
        color: black;
        font-weight: bold;
        background: rgba(255, 255, 255, 0.8);
        padding: 8px 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    </style>
    <div class="footer">This website was created by <b>Jeswant P</b> and <b>Akash V</b></div>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# ✅ Load dataset (Tamil_movies.csv)
movies = pd.read_csv('Tamil_movies.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
movie_names = sorted(movies["MovieName"].dropna().unique().tolist())

# ✅ TMDb API Key (Replace with your actual key)
TMDB_API_KEY = "8ee5ab944bdec90d5551d7b609adba61"

# ✅ Function to get IMDb link
def get_imdb_link(movie_name):
    return f"https://www.imdb.com/find?q={movie_name.replace(' ', '+')}&s=tt"

# ✅ Function to get movie poster from TMDb API
def get_movie_poster(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url).json()
    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    return None

# ✅ Streamlit UI
st.title("🎬  Movie Recommendation System")
# ✅ Movie selection via dropdown only (no text input)
selected_movie = st.selectbox(
    "Select a movie:", 
    movie_names,  # Show all movies by default
    key="movie_select"
)

# ✅ Display recommendations when a movie is selected
if selected_movie and st.button("Recommend"):
    content_based_recommendations = recommend_movies_content_based(selected_movie)
    collab_recommendations = collaborative_filtering(selected_movie)
    
    # Combine and display recommendations (your existing code)
    recommendations = list(set(content_based_recommendations + collab_recommendations))
# ✅ Create content feature for content-based filtering
movies['content'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + movies['Actor']

# ✅ Vectorization for content-based filtering
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ Content-based Recommendation function
def recommend_movies_content_based(title, num_recommendations=15):
    title = title.strip().lower()
    movies['clean_name'] = movies['MovieName'].str.strip().str.lower()

    if not movies['clean_name'].eq(title).any():
        return []

    idx = movies[movies['clean_name'] == title].index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return movies.iloc[sim_indices]['MovieName'].tolist()

# ✅ Collaborative Filtering (Matrix Factorization)
def collaborative_filtering(movie_name, num_recommendations=10):
    try:
        user_movie_ratings = pd.read_csv("Tamil_movies.csv")  # User ratings dataset
        # Ensure all necessary columns are present
        if 'MovieID' not in user_movie_ratings or 'Rating' not in user_movie_ratings:
            return []

        # Creating the rating matrix using MovieID and UserID
        rating_matrix = user_movie_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

        # Apply Singular Value Decomposition (SVD)
        U, sigma, Vt = svds(rating_matrix, k=50)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        # Convert predicted ratings to DataFrame
        predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=rating_matrix.columns)

        # Check if movie_name exists and get recommendations
        movie_id = user_movie_ratings[user_movie_ratings['MovieName'].str.lower() == movie_name.lower()]['MovieID'].values
        if movie_id.size > 0:
            movie_id = movie_id[0]
            similar_scores = predicted_ratings_df[movie_id].sort_values(ascending=False)
            return user_movie_ratings[user_movie_ratings['MovieID'].isin(similar_scores.index[:num_recommendations])]['MovieName'].tolist()
        return []
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return []

# ✅ Display recommendations
if selected_movie and st.button("Recommend"):
    content_based_recommendations = recommend_movies_content_based(selected_movie)
    collab_recommendations = collaborative_filtering(selected_movie)

    # ✅ Combine recommendations
    recommendations = list(set(content_based_recommendations + collab_recommendations))

    # ✅ Display recommendations with posters
    cols = st.columns(5)  # 5 movies per row
    for i, movie in enumerate(recommendations):
        imdb_url = get_imdb_link(movie)
        poster_url = get_movie_poster(movie)

        with cols[i % 5]:  # Arrange in horizontal rows
            if poster_url:
                st.markdown(
                    f'<a href="{imdb_url}" target="_blank">'
                    f'<img src="{poster_url}" width="150px" style="border-radius:10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);"></a>',
                    unsafe_allow_html=True
                )
            st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)  # Movie name in red

# Add custom styles
add_custom_styles()
