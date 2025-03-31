import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Function to add background image, text colors, and footer
def add_custom_styles(image_url):
    css_code = f"""
    <style>
    .stApp {{
        background-image: url({image_url});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    body, .stTextInput, .stSelectbox {{
        color: black !important;
        font-weight: bold;
    }}
    .movie-title {{
        color: red;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        margin-top: 5px;
    }}
    div.stButton > button {{
        background-color: green !important;
        color: white !important;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }}
    .footer {{
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
    }}
    </style>
    <div class="footer">This website was created by <b>Jeswant P</b> and <b>Akash V</b></div>
    """
    st.markdown(css_code, unsafe_allow_html=True)



# ✅ Load dataset
movies = pd.read_csv('Tamil_movies.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
movie_names = sorted(movies["MovieName"].dropna().unique().tolist())

# ✅ TMDb API Key (Replace with your actual key)
TMDB_API_KEY = "8ee5ab944bdec90d5551d7b609adba61"

# ✅ Function to get movie poster
def get_movie_poster(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url).json()
    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    return None

# ✅ Function to get IMDb link
def get_imdb_link(movie_name):
    return f"https://www.imdb.com/find?q={movie_name.replace(' ', '+')}&s=tt"

# ✅ Streamlit UI
st.title("🎬  Movie Recommendation System")

# ✅ Movie Search with Live Suggestions
movie_query = st.text_input("Enter a movie name:")
filtered_movies = [m for m in movie_names if movie_query.lower() in m.lower()][:10] if movie_query else []

selected_movie = st.selectbox("Select a movie:", filtered_movies) if filtered_movies else None

# ✅ Create content feature
movies['content'] = movies['Genre'] + ' ' + movies['Director'] + ' ' + movies['Actor']

# ✅ Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ Recommendation function
def recommend_movies(title, num_recommendations=15):
    title = title.strip().lower()
    movies['clean_name'] = movies['MovieName'].str.strip().str.lower()

    if not movies['clean_name'].eq(title).any():
        return []

    idx = movies[movies['clean_name'] == title].index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return movies.iloc[sim_indices]['MovieName'].tolist()

# ✅ Display recommendations
if selected_movie and st.button("Recommend"):
    recommended_movies = recommend_movies(selected_movie)

    # ✅ Display movies horizontally
    cols = st.columns(5)  # 5 movies per row

    for i, movie in enumerate(recommended_movies):
        poster_url = get_movie_poster(movie)
        imdb_url = get_imdb_link(movie)  # ✅ IMDb Link

        with cols[i % 5]:  # ✅ Arrange in horizontal rows
            if poster_url:
                st.markdown(
                    f'<a href="{imdb_url}" target="_blank">'
                    f'<img src="{poster_url}" width="150px" style="border-radius:10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);"></a>',
                    unsafe_allow_html=True
                )
            st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)  # ✅ Movie name in red
