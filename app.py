"""
Flask Application for Spam Email Prediction System
===================================================
A web-based spam email detection system using Machine Learning
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import os
import re
import string
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
MODEL_DIR = 'models'
model = None
vectorizer = None

def load_model_file(filepath):
    """Load model file based on extension (.joblib or .pkl)"""
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        return joblib.load(filepath)

def load_latest_model(prefer_pkl=False):
    """Load the latest model and vectorizer from the models directory
    
    Args:
        prefer_pkl: If True, prefer .pkl files over .joblib files
    """
    global model, vectorizer
    
    try:
        # Find all model and vectorizer files (both .joblib and .pkl)
        all_files = os.listdir(MODEL_DIR)
        
        model_files_joblib = [f for f in all_files if f.startswith('best_model_') and f.endswith('.joblib')]
        model_files_pkl = [f for f in all_files if f.startswith('best_model_') and f.endswith('.pkl')]
        
        vectorizer_files_joblib = [f for f in all_files if f.startswith('tfidf_vectorizer_') and f.endswith('.joblib')]
        vectorizer_files_pkl = [f for f in all_files if f.startswith('tfidf_vectorizer_') and f.endswith('.pkl')]
        
        # Choose file format based on preference
        if prefer_pkl and model_files_pkl and vectorizer_files_pkl:
            model_files = model_files_pkl
            vectorizer_files = vectorizer_files_pkl
        elif model_files_joblib and vectorizer_files_joblib:
            model_files = model_files_joblib
            vectorizer_files = vectorizer_files_joblib
        elif model_files_pkl and vectorizer_files_pkl:
            model_files = model_files_pkl
            vectorizer_files = vectorizer_files_pkl
        else:
            print("❌ No model files found")
            return False
        
        # Sort by timestamp (latest first)
        model_files.sort(reverse=True)
        vectorizer_files.sort(reverse=True)
        
        model_path = os.path.join(MODEL_DIR, model_files[0])
        vectorizer_path = os.path.join(MODEL_DIR, vectorizer_files[0])
        
        model = load_model_file(model_path)
        vectorizer = load_model_file(vectorizer_path)
        
        print(f"✅ Model loaded: {model_files[0]}")
        print(f"✅ Vectorizer loaded: {vectorizer_files[0]}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    return False

def clean_text(text):
    """Clean and preprocess email text"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_email(email_text):
    """Predict if an email is spam or not"""
    if model is None or vectorizer is None:
        return {
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }
    
    # Clean the text
    cleaned_text = clean_text(email_text)
    
    if len(cleaned_text) < 5:
        return {
            'success': False,
            'error': 'Email text is too short for prediction.'
        }
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    
    # Get confidence - handle models that don't support predict_proba
    confidence = None
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vectorized)[0]
            # Get the confidence for the predicted class
            if prediction == 1:
                confidence = proba[1] * 100  # Spam probability
            else:
                confidence = proba[0] * 100  # Ham probability
        elif hasattr(model, 'decision_function'):
            # For models like LinearSVC that use decision_function
            decision = model.decision_function(text_vectorized)[0]
            # Convert decision function to pseudo-probability using sigmoid
            import math
            prob = 1 / (1 + math.exp(-decision))
            confidence = (prob if prediction == 1 else (1 - prob)) * 100
    except Exception as e:
        print(f"Warning: Could not get confidence score: {e}")
        confidence = None
    
    return {
        'success': True,
        'is_spam': bool(prediction == 1),
        'prediction': 'SPAM' if prediction == 1 else 'HAM',
        'confidence': round(confidence, 1) if confidence is not None else None,
        'cleaned_text_preview': cleaned_text[:200] + '...' if len(cleaned_text) > 200 else cleaned_text
    }

# Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Prediction page"""
    result = None
    email_text = ""
    
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')
        if email_text:
            result = predict_email(email_text)
    
    return render_template('predict.html', result=result, email_text=email_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    data = request.get_json()
    
    if not data or 'email_text' not in data:
        return jsonify({
            'success': False,
            'error': 'No email text provided'
        }), 400
    
    result = predict_email(data['email_text'])
    return jsonify(result)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'timestamp': datetime.now().isoformat()
    })

# Load model on startup
with app.app_context():
    load_latest_model()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
