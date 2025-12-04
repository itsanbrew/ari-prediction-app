# ARI Prediction Application

A full-stack web application for predicting Acute Respiratory Infection (ARI) based on symptoms.

## Project Structure

```
ari_prediction_app/
├── backend-local/    # FastAPI Python backend for local hosting (optional)
│   ├── main.py       # FastAPI server
│   └── requirements.txt
├── frontend/         # Next.js React frontend with integrated API routes
│   ├── app/          # Next.js app directory
│   ├── api/          # Python serverless functions (for Vercel & local dev)
│   │   ├── predict.py
│   │   └── health.py
│   ├── lib/          # Model files (used by API routes)
│   │   ├── ari_prediction_model.pkl
│   │   └── model_metadata.json
│   ├── package.json
│   └── requirements.txt
└── README.md
```

## Setup Instructions

### Quick Start (Recommended)

The frontend includes integrated API routes, so you only need to run the frontend:

1. Navigate to the frontend directory:
```bash
cd ari_prediction_app/frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Install Python dependencies (for API routes):
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

The API routes are automatically available at:
- `http://localhost:3000/api/health` - Health check
- `http://localhost:3000/api/predict` - Prediction endpoint

### Alternative: Standalone FastAPI Backend for Local Hosting (Optional)

If you want to run the FastAPI backend separately for local development:

1. Navigate to the backend-local directory:
```bash
cd ari_prediction_app/backend-local
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

**Note:** The frontend is configured to use `/api/predict` by default. To use the standalone FastAPI backend, set the environment variable:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/predict npm run dev
```

## Usage

1. Start the frontend server (includes API routes)
2. Open `http://localhost:3000` in your browser
3. Fill in patient information (optional)
4. Check the symptoms that apply
5. Click "Predict ARI" to get the prediction
6. Use "Clear All" to reset the form

## Features

- ✅ 12 symptom checkboxes for ARI prediction
- ✅ Patient information fields (name, age, education, religion)
- ✅ Real-time ARI prediction via API
- ✅ Clear all inputs button
- ✅ Auto-clear on page refresh
- ✅ No database - all data is in-memory only

## API Endpoints

### Next.js API Routes (Integrated - Default)
- `GET /api/health` - Health check
- `POST /api/predict` - Predict ARI based on symptoms

### FastAPI Backend (Local Hosting - Optional)
- `GET /` - API status
- `GET /health` - Health check
- `POST /predict` - Predict ARI based on symptoms

## Deployment

### Vercel Deployment

The app is configured for Vercel deployment:

1. Push code to GitHub repository
2. Import the repository in Vercel dashboard
3. **Important:** Set root directory to `ari_prediction_app/frontend` in Vercel project settings
4. Vercel will automatically:
   - Detect Next.js framework
   - Install Node.js dependencies
   - Install Python dependencies (for API routes)
   - Deploy both frontend and API routes

The API routes in `frontend/api/` will be deployed as Vercel serverless functions.

#### Troubleshooting 404 Errors

If you encounter a 404 error after deployment:

1. **Verify Root Directory:** In Vercel project settings, ensure the root directory is set to `ari_prediction_app/frontend` (not the repository root)
2. **Check Build Logs:** Review the deployment logs to ensure Python dependencies are installed
3. **Verify API Routes:** The API routes should be accessible at:
   - `https://your-app.vercel.app/api/health`
   - `https://your-app.vercel.app/api/predict`
4. **Model Files:** Ensure the model files in `frontend/lib/` are committed to Git (they're large, so this may take time)

## Notes

- The model files are included in `frontend/lib/` for API route access
- All data is processed in-memory and not saved
- The form automatically clears on page refresh
- The frontend uses `/api/predict` by default (works locally and on Vercel)

