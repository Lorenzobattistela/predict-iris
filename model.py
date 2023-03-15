import requests
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import statistics
import numpy as np
from typing import List

def get_df() -> pd.DataFrame:
    iris = datasets.load_iris(as_frame=True)
    return pd.DataFrame(iris.frame)

def get_test_train(df: pd.DataFrame):
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
    pickle.dump(model, open('model.pkl', 'wb'))

def load_model() -> KNeighborsClassifier:
    return pickle.load(open('model.pkl', 'rb'))

def main():
    df = get_df()
    classifier = train_model(df)
    save_model(classifier)
    

def predict(model: KNeighborsClassifier, prompt: List[List]):
    pred = pd.DataFrame(np.array(prompt).reshape(1, -1), columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(pred)[0]
    if(prediction == 0):
        return('It is a Iris-setosa')
    elif prediction == 1:
        return('It is a Iris-versicolor')
    elif prediction == 2:
        return('It is a Iris-virginica')


