# ARI Prediction Application

A full-stack web application for predicting Acute Respiratory Infection (ARI) based on symptoms.

## Project Structure

```
ari_prediction_app/
├── backend/          # FastAPI Python backend
│   ├── main.py       # API server
│   └── requirements.txt
├── frontend/         # Next.js React frontend
│   ├── app/          # Next.js app directory
│   └── package.json
└── README.md
```

## Localhost Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd ari_prediction_app/backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure the model files are in the DiseasePrediction root directory (parent of ari_prediction_app):
   - `ari_prediction_model.pkl`
   - `model_metadata.json`
   
   The backend will automatically find these files.

5. Run the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd ari_prediction_app/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Start the backend server (port 8000)
2. Start the frontend server (port 3000)
3. Open `http://localhost:3000` in your browser
4. Fill in patient information (optional)
5. Check the symptoms that apply
6. Click "Predict ARI" to get the prediction
7. Use "Clear All" to reset the form

## Features

- ✅ 12 symptom checkboxes for ARI prediction
- ✅ Patient information fields (name, age, education, religion)
- ✅ Real-time ARI prediction via API
- ✅ Clear all inputs button
- ✅ Auto-clear on page refresh
- ✅ No database - all data is in-memory only

## API Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `POST /predict` - Predict ARI based on symptoms

## Notes

- The model files (`ari_prediction_model.pkl` and `model_metadata.json`) should be in the parent directory of the backend
- All data is processed in-memory and not saved
- The form automatically clears on page refresh

