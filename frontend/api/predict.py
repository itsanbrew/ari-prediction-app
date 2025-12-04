from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

import joblib
import numpy as np

# Load model and metadata
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'lib' / 'ari_prediction_model.pkl'
METADATA_PATH = BASE_DIR / 'lib' / 'model_metadata.json'

# Load model (cached on first import)
try:
    model = joblib.load(str(MODEL_PATH))
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        feature_columns = metadata['feature_columns']
except Exception as e:
    model = None
    error_msg = str(e)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if model is None:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Model not loaded: {error_msg}'}).encode('utf-8'))
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            symptoms = body.get('symptoms', {})
            
            # Convert symptom booleans to 0/1 array in the correct order
            symptom_values = [
                1 if symptoms.get('coryza', False) else 0,
                1 if symptoms.get('wheezing', False) else 0,
                1 if symptoms.get('sore_throat', False) else 0,
                1 if symptoms.get('nasal_congestion', False) else 0,
                1 if symptoms.get('fever', False) else 0,
                1 if symptoms.get('cough', False) else 0,
                1 if symptoms.get('chills', False) else 0,
                1 if symptoms.get('difficulty_breathing', False) else 0,
                1 if symptoms.get('coughing_up_sputum', False) else 0,
                1 if symptoms.get('shortness_of_breath', False) else 0,
                1 if symptoms.get('chest_tightness', False) else 0,
                1 if symptoms.get('vomiting', False) else 0,
            ]
            
            # Convert to numpy array and reshape for prediction
            features = np.array(symptom_values).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Get the index of ARI in the model's classes
            ari_index = list(model.classes_).index('ARI')
            ari_probability = probabilities[ari_index]
            
            # Determine confidence level
            if ari_probability >= 0.8:
                confidence = "High"
            elif ari_probability >= 0.6:
                confidence = "Moderate"
            else:
                confidence = "Low"
            
            response = {
                "prediction": prediction,
                "probability": round(float(ari_probability), 4),
                "confidence": confidence
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            import traceback
            error_response = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass
