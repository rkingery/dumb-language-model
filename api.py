import json
import torch
import torchtext
from flask import Flask, jsonify, request

from utils import generate_text, get_model, get_vocab, MAX_LEN, TEMPERATURE


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    response = request.get_json()
    text = response['text']
    max_len = response['max_len'] if response['max_len'] is not None else MAX_LEN
    temperature = response['temperature'] if response['temperature'] is not None else TEMPERATURE
    vocab = get_vocab()
    model = get_model()
    generated = generate_text(text, model, vocab, max_len=max_len, temperature=temperature)
    return jsonify({'generated_text': generated})


if __name__ == '__main__':
    app.run()
    

# To test, run flask at localhost:5000 (default port) and do

# import requests, json

# data = {
#     'text': 'This is a story about',
#     'max_len': None,
#     'temperature': None
# }

# resp = requests.post("http://127.0.0.1:5000/predict", json=data)
# output = json.loads(resp.content)

# print(output)