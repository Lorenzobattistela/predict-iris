from flask import Flask, request, jsonify, make_response
import jwt
from functools import wraps
from model import load_model, predict

app = Flask(__name__)

ai_model = load_model()

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
def predict_route():
    json = request.get_json()
    prompt = []
    for item in json:
        prompt.append(json[item])
    prediction = predict(ai_model, prompt)
    return prediction
