from flask import Flask, request, jsonify, make_response
import uuid
import jwt
import datetime
from functools import wraps
import model

app = Flask(__name__)

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'No token provided'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Signature expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        return f(data, *args, **kwargs)
    return decorated_function


@app.route("/")
def home():
    return "OK"

@token_required
@app.route("/predict", methods=['POST'])
def predict():
    return request.get_json()
