import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  

# ✅ Add custom CSS for background image, black font color, and footer
def add_custom_styles(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        body, .stTextInput, .stSelectbox, .stButton, .stMarkdown {{
            color: black !important;
            font-weight: bold;
        }}
        /* ✅ Footer Styling */
        .footer {{
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 16px;
            color: black;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.7);
            padding: 5px;
            border-radius: 10px;
        }}
        </style>
        <div class="footer">This website was created by Jeswant P and Akash V</div>
        """,
        unsafe_allow_html=True
    )

# 🔹 Background Image (Tamil Actor Collage)
background_image_url = "https://www.google.com/imgres?q=ott%20platforms%20images%20high%20quality&imgurl=https%3A%2F%2Fimg-cdn.thepublive.com%2Ffit-in%2F1200x675%2Fvnd%2Fmedia%2Fpost_banners%2Fwp-content%2Fuploads%2F2021%2F03%2FCS-Indias-top-OTT-platforms.jpg&imgrefurl=https%3A%2F%2Fwww.voicendata.com%2Findias-top-ott-platforms%2F&docid=1obpYzgES9xNPM&tbnid=NYHk0__pPtgRYM&vet=12ahUKEwjbjMigwrOMAxVRzjgGHeqMFQQQM3oECFgQAA..i&w=1200&h=630&hcb=2&ved=2ahUKEwjbjMigwrOMAxVRzjgGHeqMFQQQM3oECFgQAA"
add_custom_styles(background_image_url)

# ✅ Load dataset
movies = pd.read_csv('Tamil_movies.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
movie_names = movies["MovieName"].dropna().unique().tolist()

# ✅ Function for autocomplete suggestions
def get_suggestions(query, choices, limit=5):
    return [match[0] for match in process.extract(query, choices, limit=limit)]

# ✅ TMDb API Key (Replace with your actual key)
TMDB_API_KEY = "YOUR_TMDB_API_KEY"

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
                    f'<img src="{poster_url}" width="150px" style="border-radius:10px; box-shadow: 5px 5px 15px rgba(0,0,0,0.5);"></a>',
                    unsafe_allow_html=True
                )
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  

# ✅ Add custom CSS for background image, black font color, and footer
def add_custom_styles(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        body, .stTextInput, .stSelectbox, .stButton, .stMarkdown {{
            color: black !important;
            font-weight: bold;
        }}
        /* ✅ Footer Styling */
        .footer {{
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 16px;
            color: black;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.7);
            padding: 5px;
            border-radius: 10px;
        }}
        </style>
        <div class="footer">This website was created by Jeswant P and Akash V</div>
        """,
        unsafe_allow_html=True
    )

# 🔹 Background Image (Tamil Actor Collage)
background_image_url = "https://cdn.gulte.com/wp-content/uploads/2023/07/only-Tamil-actors-in-Tamil-films-FEFSI-kollywood.jpg"
add_custom_styles(background_image_url)

# ✅ Load dataset
movies = pd.read_csv('Tamil_movies.csv')
movies.dropna(subset=['Genre', 'Director', 'Actor'], inplace=True)
movie_names = movies["MovieName"].dropna().unique().tolist()

# ✅ Function for autocomplete suggestions
def get_suggestions(query, choices, limit=5):
    return [match[0] for match in process.extract(query, choices, limit=limit)]

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
                    f'<img src="{poster_url}" width="150px" style="border-radius:10px; box-shadow: 5px 5px 15px rgba(0,0,0,0.5);"></a>',
                    unsafe_allow_html=True
                )
