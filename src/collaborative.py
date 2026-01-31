import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_cf_model(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']], reader
    )

    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)

    return model

if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ratings.csv")
    model = train_cf_model(ratings)
    print("Collaborative Filtering model trained successfully!")
