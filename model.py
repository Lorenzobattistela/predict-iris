import requests
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import statistics
import numpy as np
from typing import List, Tuple

def get_df() -> pd.DataFrame:
    iris = datasets.load_iris(as_frame=True)
    return pd.DataFrame(iris.frame)

def get_test_train(df: pd.DataFrame) -> Tuple:
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'target'], df['target'], test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def train_model(df: pd.DataFrame) -> KNeighborsClassifier:
    x_train, x_test, y_train, y_test = get_test_train(df)
    classifier = KNeighborsClassifier()
    classifier.fit(x_train, y_train)
    return classifier

def get_labels(df: pd.DataFrame) -> List:
    return df.columns

def test_model(df: pd.DataFrame, classifier: KNeighborsClassifier) -> float:
    x_train, x_test, y_train, y_test = get_test_train(df)
    y_pred = classifier.predict(x_test)
    error = [(y_t - y_p) / 100 for y_t, y_p in zip(y_test, y_pred)]
    return 1 - statistics.mean(error)

def save_model(model: KNeighborsClassifier):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model() -> KNeighborsClassifier:
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("File model.pkl does not exist.")
    except IOError as e:
        print("An IO error occurred", e)

def main():
    df = get_df()
    classifier = train_model(df)
    save_model(classifier)
    
def get_iris_species(prediction: int) -> str:
    return ["Iris-setosa", "Iris-versicolor", "Iris-virginica"][prediction]

def preprocess(prompt: List[List]) -> pd.DataFrame:
    return pd.DataFrame(np.array(prompt).reshape(1, -1), columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

def predict(model: KNeighborsClassifier, prompt: List[List]) -> str:
    assert len(prompt) == 4, "Length format is incorrect. It should be a 4 float list."
    pred = preprocess(prompt=prompt)
    prediction = model.predict(pred)[0]
    return get_iris_species(prediction=prediction)
