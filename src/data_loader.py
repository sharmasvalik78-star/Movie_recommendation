import pandas as pd

def load_data():
    ratings = pd.read_csv("data/raw/ratings.csv")
    movies = pd.read_csv("data/raw/movies.csv")
    return ratings, movies

if __name__ == "__main__":
    ratings, movies = load_data()
    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
