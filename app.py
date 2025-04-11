from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get("sentence")
    result = classifier(sentence)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
