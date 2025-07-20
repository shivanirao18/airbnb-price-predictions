from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def train():
    df = pd.read_csv("../data/processed/airbnb_cleaned.csv")
    X = df.drop("price", axis=1)
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "../models/airbnb_model.pkl")

if __name__ == '__main__':
    train()