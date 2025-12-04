# ARI Prediction Application

A full-stack web application for predicting Acute Respiratory Infection (ARI) based on symptoms. This application runs locally and is designed for localhost development.

## Project Structure

```
ari_prediction_app/
├── backend/          # FastAPI Python backend
│   ├── main.py       # FastAPI server
│   ├── ari_prediction_model.pkl
│   ├── model_metadata.json
│   └── requirements.txt
├── frontend/         # Next.js React frontend
│   ├── app/          # Next.js app directory
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── package.json
│   └── tsconfig.json
├── start_backend.sh  # Script to start backend
├── start_frontend.sh # Script to start frontend
└── README.md
```

## Setup Instructions

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

4. Run the FastAPI server:
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

### Using the Start Scripts

Alternatively, you can use the provided shell scripts:

**Start Backend:**
```bash
cd ari_prediction_app
./start_backend.sh
```

**Start Frontend:**
```bash
cd ari_prediction_app
./start_frontend.sh
```

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
- ✅ No database - all data is processed in-memory only

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /` - API status
- `GET /health` - Health check
- `POST /predict` - Predict ARI based on symptoms

## Notes

- The model files (`ari_prediction_model.pkl` and `model_metadata.json`) are located in the `backend/` directory
- All data is processed in-memory and not saved
- The form automatically clears on page refresh
- The frontend is configured to connect to `http://localhost:8000/predict` by default

## Requirements

- Python 3.12+ (recommended)
- Node.js 18+ and npm
- All Python dependencies listed in `backend/requirements.txt`
- All Node.js dependencies listed in `frontend/package.json`
