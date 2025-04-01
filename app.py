import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np

# ========== FUNCTION DEFINITIONS ==========

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
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">
        This website was created by <b>Jeswant P</b> and <b>Akash V</b>
    </div>
    """
    st.markdown(css_code, unsafe_allow_html=True)

def get_imdb_link(movie_name):
    return f"https://www.imdb.com/find?q={movie_name.replace(' ', '+')}&s=tt"

def get_movie_poster(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url).json()
    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    return None

def search_movies_by_genre(genre):
    genre = genre.strip().lower()
    filtered_movies = movies[movies['Genre'].str.lower().str.contains(genre, na=False)]
    return filtered_movies["MovieName"].tolist()

# ========== DATA LOADING & INITIALIZATION ==========
movies = pd.read_csv('Tamil_movies.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
genres = sorted(set(
    genre.strip().lower() for sublist in movies["Genre"].dropna().str.split(',') for genre in sublist
))
TMDB_API_KEY = "8ee5ab944bdec90d5551d7b609adba61"

# ========== STREAMLIT UI ==========
st.title("🎬 Movie Recommendation System")

# Genre selection
genre_selected = st.selectbox("Select a genre:", genres, key="genre_select")

if st.button("Search Movies", key="search_button"):
    genre_movies = search_movies_by_genre(genre_selected)
    
    cols = st.columns(5)
    for i, movie in enumerate(genre_movies):
        imdb_url = get_imdb_link(movie)
        poster_url = get_movie_poster(movie)
        with cols[i % 5]:
            if poster_url:
                st.markdown(
                    f'<a href="{imdb_url}" target="_blank">'
                    f'<img src="{poster_url}" width="150px" style="border-radius:10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);"></a>',
                    unsafe_allow_html=True
                )
            st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)

# Add custom styles (must be last)
add_custom_styles()
