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
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/150x225?text=No+Image"
    return "https://via.placeholder.com/150x225?text=No+Image"



def search_movies_by_genre(genre):
    genre = genre.strip().lower()
    filtered_movies = movies[movies['Genre'].str.lower().str.split(',').apply(lambda x: genre in [g.strip() for g in x] if isinstance(x, list) else False)]
    return filtered_movies["MovieName"].tolist()


def recommend_movies_content_based(title, num_recommendations=10):
    title = title.strip().lower()
    movies['clean_name'] = movies['MovieName'].str.strip().str.lower()

    if not movies['clean_name'].eq(title).any():
        return []
    
    idx = movies[movies['clean_name'] == title].index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    # Ensure recommended movies belong to the same genre as the input movie
    input_movie_genre = movies.loc[idx, "Genre"]
    recommended_movies = movies.iloc[sim_indices]
    filtered_movies = recommended_movies[recommended_movies['Genre'].str.contains(input_movie_genre, case=False, na=False)]

    return filtered_movies['MovieName'].tolist()


def collaborative_filtering(movie_name, num_recommendations=10):
    try:
        user_movie_ratings = pd.read_csv("Tamil_movies2.csv")
        if 'MovieID' not in user_movie_ratings or 'Rating' not in user_movie_ratings:
            return []
        rating_matrix = user_movie_ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        U, sigma, Vt = svds(rating_matrix, k=50)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=rating_matrix.columns)
        movie_id = user_movie_ratings[user_movie_ratings['MovieName'].str.lower() == movie_name.lower()]['MovieID'].values
        if movie_id.size > 0:
            movie_id = movie_id[0]
            similar_scores = predicted_ratings_df[movie_id].sort_values(ascending=False)
            return user_movie_ratings[user_movie_ratings['MovieID'].isin(similar_scores.index[:num_recommendations])]['MovieName'].tolist()
        return []
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return []

# ========== DATA LOADING & INITIALIZATION ==========
movies = pd.read_csv('Tamil_movies2.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
genres = sorted(set(
    genre.strip().lower() for sublist in movies["Genre"].dropna().str.split(',') for genre in sublist
))
TMDB_API_KEY = "8ee5ab944bdec90d5551d7b609adba61"

movies['content'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + movies['Actor']
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ========== STREAMLIT UI ==========
st.title("🎬 Movie Recommendation System")

# Genre selection
genre_selected = st.selectbox("Select a genre:", genres, key="genre_select")

if st.button("Search Movies", key="search_button"):
    genre_movies = search_movies_by_genre(genre_selected)
    recommendations = []
    
    # Generate recommendations
    for movie in genre_movies:
        recommendations.extend(recommend_movies_content_based(movie, 3))
        recommendations.extend(collaborative_filtering(movie, 3))
    
    recommendations = list(set(recommendations))  # Remove duplicates
    
    # Filter out movies with no posters
    valid_recommendations = []
    for movie in recommendations:
        poster_url = get_movie_poster(movie)
        if poster_url and "placeholder.com" not in poster_url:  # Ensure a valid poster exists
            valid_recommendations.append((movie, poster_url))

    # Display only movies with valid posters
    cols = st.columns(5)
   for i, movie in enumerate(recommendations):
    imdb_url = get_imdb_link(movie)
    poster_url = get_movie_poster(movie)  # Fetch movie poster
    
    with cols[i % 5]:  # Ensure proper indentation inside loop
        st.markdown(
            f'<a href="{imdb_url}" target="_blank">'
            f'<img src="{poster_url}" width="150px" style="border-radius:10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);"></a>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)




# Add custom styles (must be last)
add_custom_styles()
