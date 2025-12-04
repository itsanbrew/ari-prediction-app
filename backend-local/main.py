from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load the model and metadata
# Model files should be in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ari_prediction_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

# Load model
model = joblib.load(MODEL_PATH)

# Load metadata to get feature order
import json
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
    feature_columns = metadata['feature_columns']

app = FastAPI(title="ARI Prediction API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for symptoms
class SymptomInput(BaseModel):
    coryza: bool = False
    wheezing: bool = False
    sore_throat: bool = False
    nasal_congestion: bool = False
    fever: bool = False
    cough: bool = False
    chills: bool = False
    difficulty_breathing: bool = False
    coughing_up_sputum: bool = False
    shortness_of_breath: bool = False
    chest_tightness: bool = False
    vomiting: bool = False

# Request model for patient info (optional, not used in prediction)
class PatientInfo(BaseModel):
    name: str = ""
    age: int = 0
    education: str = ""
    religion: str = ""

# Combined request model
class PredictionRequest(BaseModel):
    symptoms: SymptomInput
    patient_info: PatientInfo = PatientInfo()

# Response model
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str

@app.get("/")
def read_root():
    return {"message": "ARI Prediction API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_ari(request: PredictionRequest):
    """
    Predict ARI based on symptoms
    """
    # Convert symptom booleans to 0/1 array in the correct order
    symptom_values = [
        1 if request.symptoms.coryza else 0,
        1 if request.symptoms.wheezing else 0,
        1 if request.symptoms.sore_throat else 0,
        1 if request.symptoms.nasal_congestion else 0,
        1 if request.symptoms.fever else 0,
        1 if request.symptoms.cough else 0,
        1 if request.symptoms.chills else 0,
        1 if request.symptoms.difficulty_breathing else 0,
        1 if request.symptoms.coughing_up_sputum else 0,
        1 if request.symptoms.shortness_of_breath else 0,
        1 if request.symptoms.chest_tightness else 0,
        1 if request.symptoms.vomiting else 0,
    ]
    
    # Convert to numpy array and reshape for prediction
    features = np.array(symptom_values).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get the index of ARI in the model's classes
    # sklearn orders classes alphabetically, so 'ARI' comes before 'non-ARI'
    ari_index = list(model.classes_).index('ARI')
    ari_probability = probabilities[ari_index]
    
    # Determine confidence level
    if ari_probability >= 0.8:
        confidence = "High"
    elif ari_probability >= 0.6:
        confidence = "Moderate"
    else:
        confidence = "Low"
    
    return PredictionResponse(
        prediction=prediction,
        probability=round(float(ari_probability), 4),
        confidence=confidence
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

