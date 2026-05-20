# Eye Disease Prediction

Deep learning-powered web application for detecting retinal diseases from OCT (Optical Coherence Tomography) scans. Built with TensorFlow MobileNetV3 and FastAPI.

## Detected Conditions

| Class | Description |
|-------|-------------|
| **CNV** | Choroidal Neovascularization |
| **DME** | Diabetic Macular Edema |
| **DRUSEN** | Drusen deposits (Early AMD) |
| **NORMAL** | Healthy retina |

## Model Performance

- **Architecture**: MobileNetV3Large (transfer learning)
- **Test Accuracy**: 94.76%
- **Dataset**: ~109,000 OCT images (4 classes)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m uvicorn app:app --reload

# Open http://localhost:8000
```

## Project Structure

```
├── app.py                  # FastAPI web application
├── app_streamlit.py        # Legacy Streamlit app
├── recommendation.py       # Medical recommendation data
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html          # Frontend (single-page, dark theme)
├── Trained_Model.h5        # Trained model weights (64 MB)
├── Trained_Model.keras     # Trained model (Keras format, 22 MB)
├── Training_model.ipynb    # Model training notebook
└── Model_Prediction.ipynb  # Inference notebook
```

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **ML**: TensorFlow / Keras (MobileNetV3Large)
- **Frontend**: Vanilla HTML/CSS/JS (single file, no frameworks)

## License

MIT
