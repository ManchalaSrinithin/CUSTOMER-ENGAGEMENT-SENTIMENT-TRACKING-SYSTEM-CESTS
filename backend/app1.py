import os
import logging
import sys
import tempfile
import pickle
import numpy as np
import tensorflow as tf
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from groq import Groq
from dotenv import load_dotenv
whisper_model = whisper.load_model("base")

import requests  # Make sure to import the requests module


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set environment variables to disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Disable tensorflow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Groq client with environment variable
GROQ_API_KEY = os.environ('grok_api_key')
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Model parameters
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 200


class TextAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        try:
            self.sia = SentimentIntensityAnalyzer()
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            logger.error(f"Error initializing VADER: {str(e)}")
            self.sia = None

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ''
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ''

    def build_model(self):
        """Build the LSTM model"""
        try:
            input_layer = Input(shape=(MAX_LEN,))
            embedding = Embedding(MAX_WORDS, EMBEDDING_DIM)(input_layer)
            lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
            lstm = Dropout(0.2)(lstm)
            lstm = Bidirectional(LSTM(32))(lstm)
            lstm = Dropout(0.2)(lstm)
            dense = Dense(16, activation='relu')(lstm)
            output = Dense(1, activation='sigmoid')(dense)
            model = Model(inputs=input_layer, outputs=output)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return None

    def generate_summary(self, text):
        """Generate summary using Groq API"""
        if not groq_client:
            return "Summary generation unavailable - API key not configured"
            
        try:
            prompt = f"Please provide a brief summary of the following text: {text}"
            
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "system", "content": "You are a helpful assistant that provides concise summaries."},
                          {"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                return "Error: Invalid API response"
                
            summary = response.choices[0].message.content
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def predict_sentiment(self, text):
        """Predict sentiment for a given text"""
        if not text:
            return {'sentiment': 'neutral', 'score': 0.5, 'error': 'Empty text provided'}

        try:
            if self.model is None or self.tokenizer is None:
                if self.sia is None:
                    return {'sentiment': 'neutral', 'score': 0.5, 'error': 'Sentiment analyzer not initialized'}
                    
                vader_scores = self.sia.polarity_scores(text)
                combined_score = (vader_scores['compound'] + 1) / 2
                sentiment = 'neutral'
                if combined_score < 0.4:
                    sentiment = 'negative'
                elif combined_score > 0.6:
                    sentiment = 'positive'
                return {'sentiment': sentiment, 'score': float(combined_score), 'vader_scores': vader_scores}
            
            cleaned_text = self.clean_text(text)
            vader_scores = self.sia.polarity_scores(text)
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
            prediction = self.model.predict(padded_sequence, verbose=0)[0][0]
            combined_score = (prediction * 0.3 + (vader_scores['compound'] + 1) / 2 * 0.7)
            
            sentiment = 'neutral'
            if combined_score < 0.4:
                sentiment = 'negative'
            elif combined_score > 0.6:
                sentiment = 'positive'
                
            return {'sentiment': sentiment, 'score': float(combined_score), 'model_score': float(prediction), 'vader_scores': vader_scores}
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'score': 0.5, 'error': str(e)}

    def analyze_text(self, text):
        """Complete text analysis including sentiment and summary"""
        if not text:
            return {'error': 'No text provided', 'text': '', 'summary': '', 'sentiment': 'neutral', 'score': 0.5}

        try:
            sentiment_result = self.predict_sentiment(text)
            summary = self.generate_summary(text)
            
            return {'text': text, 'summary': summary, **sentiment_result}
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {'text': text, 'error': str(e), 'summary': "Error generating summary", 'sentiment': 'neutral', 'score': 0.5}

    def train(self, texts, labels, validation_split=0.2):
        """Train the sentiment analysis model"""
        if not texts or not labels:
            raise ValueError("Empty training data provided")

        try:
            cleaned_texts = [self.clean_text(text) for text in texts]
            self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(cleaned_texts)
            sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
            padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
            labels = np.array(labels)
            
            self.model = self.build_model()
            if self.model is None:
                raise ValueError("Failed to build model")
                
            history = self.model.fit(
                padded_sequences, labels,
                epochs=10,
                validation_split=validation_split,
                callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
                verbose=1
            )
            return history
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save(self, model_path='sentiment_model.h5', tokenizer_path='tokenizer.pickle'):
        """Save the model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            self.model.save(model_path)
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Model and tokenizer saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, model_path='sentiment_model.h5', tokenizer_path='tokenizer.pickle'):
        """Load the model and tokenizer"""
        try:
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                self.model = tf.keras.models.load_model(model_path)
                with open(tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                logger.info("Model and tokenizer loaded successfully")
                return True
            else:
                logger.warning("Model or tokenizer files not found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


# Initialize the text analyzer
text_analyzer = TextAnalyzer()


@app.route('/analyze', methods=['POST'])
def analyze_text_route():
    """API endpoint for text analysis"""
    try:
        data = request.json
        text = data.get('text', '')
        result = text_analyzer.analyze_text(text)
        print(result)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 400



@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    """API endpoint for audio transcription and sending the transcription to analyze"""
    # Add FFmpeg's directory to PATH
    os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-master-latest-win64-gpl\bin"

    try:
        print("Hello")
        print("Request files:", request.files)  # Log the request.files to see what is received
        
        if 'audio' not in request.files:
            print("No 'audio' file found in request.")
            return jsonify({'error': 'No audio file uploaded'}), 400
        
        audio_file = request.files['audio']
        print(f"Received file: {audio_file.filename}")
        
        # Save the audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name
        print(f"File saved to temporary path: {temp_file_path}")
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(temp_file_path)
        transcription = result['text']
        print(f"Transcription result: {transcription}")
        
        # Send transcription to analyze route
        analyze_url = 'http://127.0.0.1:5000/analyze'  # This is the local URL of the analyze route
        analyze_response = requests.post(analyze_url, json={'text': transcription})
        
        if analyze_response.status_code == 200:
            analysis_result = analyze_response.json()
            print(f"Analysis result: {analysis_result}")
        else:
            return jsonify({'error': 'Failed to analyze transcription'}), 400
        
        # Cleanup temporary file
        os.remove(temp_file_path)
        
        return jsonify({'transcription': transcription, 'analysis_result': analysis_result})
    
    except Exception as e:
        print(f"Error in transcribe_audio_route: {e}")
        return jsonify({'error': str(e)}), 400
@app.route('/train', methods=['POST'])
def train_model_route():
    """API endpoint to train the sentiment analysis model"""
    try:
        data = request.json
        texts = data.get('texts', [])
        labels = data.get('labels', [])
        
        if not texts or not labels:
            return jsonify({'error': 'Training data is missing'}), 400
        
        # Train the model
        text_analyzer.train(texts, labels)
        
        # Save the model and tokenizer
        text_analyzer.save()
        
        return jsonify({'message': 'Model trained and saved successfully'})
    except Exception as e:
        logger.error(f"Error in /train endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/audio', methods=['POST'])
def audio_analysis_route():
    """API endpoint for audio file analysis with improved error handling"""
    temp_file_path = None
    
    try:
        # Validate request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file uploaded'}), 400
            
        audio_file = request.files['audio']
        if not audio_file.filename:
            logger.error("Empty filename received")
            return jsonify({'error': 'Empty file submitted'}), 400
        
        logger.info(f"Processing audio file: {audio_file.filename}")
        
        # Save file with proper extension
        file_extension = audio_file.filename.rsplit('.', 1)[-1].lower()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                audio_file.save(temp_file.name)
                temp_file_path = temp_file.name
                logger.info(f"Audio saved to temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Transcribe audio
        try:
            result = whisper_model.transcribe(temp_file_path)
            transcription = result['text']
            logger.info("Transcription completed successfully")
            
            if not transcription.strip():
                logger.warning("No speech detected in audio")
                return jsonify({'error': 'No speech detected in audio'}), 400
                
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return jsonify({'error': 'Failed to transcribe audio'}), 500
        
        # Analyze transcribed text
        try:
            analysis_result = text_analyzer.analyze_text(transcription)
            logger.info("Text analysis completed")
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return jsonify({'error': 'Failed to analyze transcription'}), 500
        
        return jsonify({
            'transcription': transcription,
            'analysis_result': analysis_result
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in audio analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {str(e)}")
@app.route('/upload', methods=['POST'])
def upload_file():
    """Simple file upload test"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    audio_file = request.files['audio']
    audio_file.save(f'./{audio_file.filename}')
    return jsonify({'message': 'File uploaded successfully'}), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
