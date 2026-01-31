import pandas as pd

def preprocess_movies(movies):
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['genres'] = movies['genres'].fillna('')
    return movies

if __name__ == "__main__":
    movies = pd.read_csv("data/raw/movies.csv")
    movies = preprocess_movies(movies)
    print(movies[['title', 'genres']].head())
