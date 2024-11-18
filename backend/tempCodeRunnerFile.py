import pickle
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained LSTM model and tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    sentences = text.split('.')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def identify_speaker(sentence):
    customer_keywords = ['problem', 'issue', 'concern', 'frustrated', 'help', 'complain']
    if any(keyword in sentence.lower() for keyword in customer_keywords):
        return 'customer'
    return 'agent'

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if 'file' in request.files:
        file = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        with open(file_path, 'r') as f:
            text = f.read()
    elif 'text' in request.form:
        text = request.form['text']
    else:
        return jsonify({'error': 'Please provide either text or a file to analyze.'}), 400

    sentences = preprocess_text(text)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    customer_sentiment = []
    positive_points = []
    negative_points = []

    for i, seq in enumerate(padded_sequences):
        sentiment_score = model.predict(np.array([seq]))[0][0]
        speaker = identify_speaker(sentences[i])
        if speaker == 'customer':
            customer_sentiment.append(sentiment_score)
            if sentiment_score < 0.4:
                negative_points.append(sentences[i])
            elif sentiment_score > 0.6:
                positive_points.append(sentences[i])

    overall_sentiment_score = float(np.mean(customer_sentiment)) if customer_sentiment else 0.5
    overall_sentiment = 'negative' if overall_sentiment_score < 0.4 else 'neutral' if overall_sentiment_score < 0.6 else 'positive'
    overall_confidence = 1 - abs(overall_sentiment_score - 0.5) / 0.5

    return jsonify({
        'overall_sentiment': overall_sentiment,
        'overall_sentiment_score': overall_sentiment_score,
        'overall_confidence': overall_confidence,
        'positive_points': positive_points,
        'negative_points': negative_points
    })

if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)