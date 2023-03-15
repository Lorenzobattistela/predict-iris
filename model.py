import requests
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import statistics

def get_df() -> pd.DataFrame:
    iris = datasets.load_iris(as_frame=True)
    return pd.DataFrame(iris.data)

def get_test_train(df: pd.DataFrame):
    df.reset_index(inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df, df['index'], test_size=0.2, random_state=42)
    df.drop('index', axis=1, inplace=True)
    return x_train, x_test, y_train, y_test

def train_model(df: pd.DataFrame) -> KNeighborsClassifier:
    x_train, x_test, y_train, y_test = get_test_train(df)
    classifier = KNeighborsClassifier()
    classifier.fit(x_train, y_train)
    return classifier

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
    success_rate = test_model(df=df, classifier=classifier) * 100
    print(f'Success rate: {success_rate}%')


if __name__ == '__main__':
    main()

