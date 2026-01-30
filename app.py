import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.tmdb_helper import get_movie_by_tmdb_id, get_movie_trailer, get_poster_path

# ---------------- LOAD TMDB DATASET ----------------
@st.cache_data
def load_tmdb_movies():
    movies = pd.read_csv("artifacts/tmdb_movies.csv")

    # 🔥 Quality filters (VERY IMPORTANT)
    movies = movies[movies["popularity"] > 10]
    movies = movies[movies["rating"] > 4]

    # Drop duplicates & reset index
    movies = movies.drop_duplicates(subset="title")
    movies = movies.reset_index(drop=True)

    return movies

@st.cache_data
def build_tmdb_model(_movies):
    # Load pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Ensure no NaN values (VERY IMPORTANT)
    overview = _movies["overview"].fillna("").astype(str)
    genres = _movies["genre_ids"].fillna("").astype(str)

    text = overview + " " + genres + " " + genres + " " + genres

    # Generate embeddings
    embeddings = model.encode(text.tolist(), show_progress_bar=True)

    return cosine_similarity(embeddings)


def recommend_movies(title, movies, sim, top_n=5):
    idx = movies[movies["title"] == title].index[0]

    sim_scores = list(enumerate(sim[idx]))

    results = []

    for i, content_score in sim_scores:
        if i == idx:
            continue

        # --- genre guard (IMPORTANT FIX) ---
        selected_genres = set(str(movies.iloc[idx]["genre_ids"]).split())
        candidate_genres = set(str(movies.iloc[i]["genre_ids"]).split())

        if not selected_genres.intersection(candidate_genres):
            continue


        popularity = movies.iloc[i]["popularity"]
        rating = movies.iloc[i]["rating"]

        # Normalize popularity & rating
        popularity_norm = popularity / movies["popularity"].max()
        rating_norm = rating / 10.0

        # 🔥 Hybrid score
        final_score = (
            0.6 * content_score +
            0.25 * popularity_norm +
            0.15 * rating_norm
        )

        results.append((i, final_score))

    # Sort by final hybrid score
    results = sorted(results, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in results[:top_n]]
    return movies.iloc[top_indices]
def get_explanation(selected_movie, recommended_movie):
    reasons = []

    # Genre-based explanation
    selected_genres = set(str(selected_movie["genre_ids"]).split())
    rec_genres = set(str(recommended_movie["genre_ids"]).split())
    common_genres = selected_genres.intersection(rec_genres)

    if common_genres:
        reasons.append(
            "Similar genres: " + ", ".join(list(common_genres)[:2])
        )

    # Popularity-based explanation
    if recommended_movie["popularity"] > selected_movie["popularity"]:
        reasons.append("Highly popular on TMDB")

    if not reasons:
        reasons.append("Similar content and themes")

    return " • ".join(reasons)



# ---------------- STREAMLIT UI ----------------
st.set_page_config(layout="wide")
st.title("🎬 TMDB Movie Recommendation System")

movies = load_tmdb_movies()
cosine_sim = build_tmdb_model(movies)

movie_name = st.selectbox("Choose a movie:", movies["title"])

if st.button("Recommend"):
    recommendations = recommend_movies(movie_name, movies, cosine_sim)

    cols = st.columns(5)
    for col, (_, row) in zip(cols, recommendations.iterrows()):
        with col:
            movie_data = get_movie_by_tmdb_id(row["tmdbId"])

            poster = get_poster_path(movie_data.get("poster_path")) if movie_data else None

            if poster:
                st.image(poster, width=250)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Poster", width=250)

            st.markdown(f"**{row['title']}**")

            selected_movie = movies[movies["title"] == movie_name].iloc[0]

            explanation = get_explanation(selected_movie, row)

            st.caption(f"Because you watched **{movie_name}**")
            st.caption(explanation)

            st.caption(f"⭐ {row['rating']}")

            if movie_data and movie_data.get("overview"):
                st.write(movie_data["overview"][:120] + "...")

            trailer = get_movie_trailer(row["tmdbId"])
            if trailer:
                st.markdown(f"[▶ Watch Trailer]({trailer})")
