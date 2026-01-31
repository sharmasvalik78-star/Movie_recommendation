import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONTENT BASED ----------
def build_content_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# ---------- COLLABORATIVE ----------
def train_cf_model(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']], reader
    )
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

# ---------- HYBRID ----------
def hybrid_recommend(user_id, movie_title, movies, ratings, alpha=0.7, top_n=5):
    movies = movies.reset_index(drop=True)

    # Content-based
    cosine_sim = build_content_model(movies)
    idx = movies[movies['title'] == movie_title].index[0]
    content_scores = list(enumerate(cosine_sim[idx]))

    # Collaborative
    cf_model = train_cf_model(ratings)

    final_scores = []

    for i, cb_score in content_scores:
        movie_id = movies.iloc[i]['movieId']
        cf_score = cf_model.predict(user_id, movie_id).est
        final_score = alpha * cf_score + (1 - alpha) * cb_score
        final_scores.append((i, final_score))

    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movies.iloc[i[0]]['title'] for i in final_scores[1:top_n+1]]

    return top_movies

# ---------- RUN ----------
if __name__ == "__main__":
    movies = pd.read_csv("data/raw/movies.csv")
    ratings = pd.read_csv("data/raw/ratings.csv")

    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

    recommendations = hybrid_recommend(
        user_id=1,
        movie_title="Toy Story (1995)",
        movies=movies,
        ratings=ratings
    )

    print("Hybrid Recommendations:")
    for movie in recommendations:
        print(movie)
