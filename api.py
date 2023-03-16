from flask import Flask, request, jsonify, make_response
from typing import List, Dict
import jwt
from functools import wraps
from model import load_model, predict
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)

ai_model = load_model()

def generate_token() -> str:
    credentials = get_credentials()
    secret = os.getenv("SECRET")
    return jwt.encode(credentials, secret)

def get_credentials() -> Dict:
    return {"login": os.getenv("LOGIN"), "password": os.getenv("PASSWORD")}

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'No token provided'}), 401
        try:
            data = jwt.decode(token, os.getenv("SECRET"), algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Signature expired. Please log in again.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token. Please log in again.'}), 401
        return f(data, *args, **kwargs)
    return decorated_function

@app.route("/")
def home():
    return "OK"

@app.route("/predict", methods=['POST'])
@token_required
def predict_route(x):
    data = request.get_json()
    prompt = format_data(data=data)

    if not is_prompt_format_correct(prompt=prompt):
        return jsonify({"error": "prompt format is not correct"})

    prediction = predict(ai_model, prompt)
    return jsonify({"prediction": prediction})

@app.route("/generate-token", methods=['POST'])
def get_token():
    correct_credentials = get_credentials()
    data = request.get_json()
    print(data)
    if data != correct_credentials:
        return jsonify(
            {"error": "Wrong credentials. Format should be {'login': 'login', 'password', 'password'}"}
        )
    return jsonify({"token": generate_token()})


def is_prompt_format_correct(prompt: List) -> bool:
    return True if len(prompt) == 4 else False

def format_data(data: Dict) -> List:
    try:
        return [float(data[item]) for item in data]
    except ValueError:
        return []
        
