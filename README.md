# Predict Iris

This is a simple Data Science project using the Iris Dataset.
The main goal of the project is to simulate a real life project where the model is exposed
to public via API. It uses a simple Flask API to expose our model and make predictions about Iris based on four attributes passed on the request.

The API is secured by JWT token authentication and generation using a login defined on ENV.

The model is simply a K-Nearest-Neighbors algorithm using scikit-learn that classificates input based on its nearest neighbors.

If you have any suggestions on code improvement, feel free to open an Issue or Pull Request
