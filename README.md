# 🎬 AI-Powered Movie Recommendation System

A modern **movie recommendation web application** built using **Python, Streamlit, and TMDB data**.  
The system recommends movies using a **hybrid approach** that combines **content similarity, popularity, and ratings**, and displays **posters, trailers, and explanations** in a clean UI.

---

## 🚀 Features

- 🔍 Select a movie and get personalized recommendations  
- 🧠 Hybrid recommendation system:
  - Content-based filtering (TF-IDF + Cosine Similarity)
  - Popularity and rating-based weighting
  - Genre-intersection constraint to avoid unrelated results
- 🖼️ Movie posters from TMDB  
- ▶️ Trailer links  
- ⭐ Movie ratings and popularity  
- 💬 “Because you watched…” explanations  
- 🎨 Dark-themed, responsive Streamlit UI  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- TMDB API  

---

## 📊 Dataset

- TMDB (The Movie Database) movie metadata  
- Fields used:
  - title
  - overview
  - genre_ids
  - popularity
  - rating
  - tmdbId  

---

## 🧠 Recommendation Logic

1. **Content Similarity**
   - Movie overview and genres are vectorized using TF-IDF
   - Cosine similarity is used to find similar movies

2. **Hybrid Scoring**
