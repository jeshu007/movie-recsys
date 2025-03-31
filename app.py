import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  

# ✅ Add custom CSS for background image & black font color
def add_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* ✅ Change font color to white */
        body, .stTextInput, .stSelectbox, .stButton, .stMarkdown {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔹 Replace with a high-quality Tamil actor image URL
background_image_url = "https://cdn.gulte.com/wp-content/uploads/2023/07/only-Tamil-actors-in-Tamil-films-FEFSI-kollywood.jpg"
add_bg_from_url(background_image_url)

# ✅ Load dataset
movies = pd.read_csv('Tamil_movies.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
movie_names = movies["MovieName"].dropna().unique().tolist()

# ✅ Function for autocomplete suggestions
def get_suggestions(query, choices, limit=5):
    suggestions = process.extract(query, choices, limit=limit)
    return [match[0] for match in suggestions]

# ✅ TMDb API Key (Replace with your actual key)
TMDB_API_KEY = "8ee5ab944bdec90d5551d7b609adba61"

# ✅ Function to get movie poster
def get_movie_poster(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url).json()
    if response['results']:
        poster_path = response['results'][0].get('poster_path', None)
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# ✅ Function to get IMDb/Wikipedia link
def get_movie_link(movie_name):
    imdb_url = f"https://www.imdb.com/find?q={movie_name.replace(' ', '+')}&s=tt"
    wiki_url = f"https://en.wikipedia.org/wiki/{movie_name.replace(' ', '_')}"
    return imdb_url, wiki_url

# ✅ Streamlit UI
st.title("🎬 Tamil Movie Recommendation System")

# ✅ User input with autocomplete
movie_query = st.text_input("Enter a movie name:")

selected_movie = None
if movie_query:
    suggestions = get_suggestions(movie_query, movie_names)
    selected_movie = st.selectbox("Did you mean:", suggestions)

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
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return movies.iloc[sim_indices]['MovieName'].tolist()

# ✅ Display recommendations
if selected_movie and st.button("Recommend"):
    recommended_movies = recommend_movies(selected_movie)

    # ✅ Display movies horizontally
    cols = st.columns(5)  # 5 movies per row

    for i, movie in enumerate(recommended_movies):
        poster_url = get_movie_poster(movie)
        imdb_url, wiki_url = get_movie_link(movie)  # ✅ Get links

        with cols[i % 5]:  # ✅ Arrange in horizontal rows
            if poster_url:
                st.markdown(
                    f'<a href="{imdb_url}" target="_blank">'
                    f'<img src="{poster_url}" width="150px" style="border-radius:10px;"></a>',
                    unsafe_allow_html=True
                )
