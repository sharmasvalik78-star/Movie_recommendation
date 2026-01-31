import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_content_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_movies(title, movies, cosine_sim, top_n=5):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

if __name__ == "__main__":
    movies = pd.read_csv("data/raw/movies.csv")
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

    cosine_sim = build_content_model(movies)
    recommendations = recommend_movies("Toy Story (1995)", movies, cosine_sim)

    print("Recommended movies:")
    print(recommendations)
