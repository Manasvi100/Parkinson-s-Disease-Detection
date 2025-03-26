from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Allows frontend to access backend from different ports
import joblib
import numpy as np
import librosa
import xgboost as xgb  
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index1')
def index1():
    return render_template("index1.html")

@app.route('/questionnaire')
def questionnaire():
    return render_template("questionnaire.html")

@app.route('/parkinsonsinfo')
def parkinsonsinfo():
    return render_template("parkinsonsinfo.html")  # Page for Parkinson’s Disease Information

@app.route('/hospitals')
def hospitals():
    return render_template("hospitals.html")  # New hospitals page

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
model_path = os.path.join(BASE_DIR, "My_New_Flask_app_zipped", "My_New_Flask_app", "xgb_parkinsons_model.pkl")
scaler_path = os.path.join(BASE_DIR, "My_New_Flask_app_zipped", "My_New_Flask_app", "scaler.pkl")


# Load trained model and scaler
try:
    model = joblib.load("xgb_parkinsons_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model, scaler = None, None

def extract_features_from_audio(file):
    try:
        y, sr = librosa.load(file, sr=None)
        print("Audio loaded successfully")

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, fmin=sr * 0.01), axis=1)

        features = np.hstack([mfccs, chroma, spectral_contrast])
        print("Extracted features shape:", features.shape)

        expected_features = scaler.n_features_in_
        print("Scaler expects:", expected_features)

        if features.shape[0] < expected_features:
            features = np.pad(features, (0, expected_features - features.shape[0]), mode='constant')
        elif features.shape[0] > expected_features:
            features = features[:expected_features]

        features = features.reshape(1, -1)
        features = scaler.transform(features)
        print("Features transformed successfully")
        return features

    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return str(e)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files["file"]

        # Extract features
        features = extract_features_from_audio(file)
        if isinstance(features, str):
            return jsonify({"error": f"Feature Extraction Error: {features}"})

        # Convert to XGBoost DMatrix
        dmatrix_features = xgb.DMatrix(features)

        # Get prediction probability
        probability = model.predict(dmatrix_features)[0]

        result = "Parkinson's Detected 🟠" if probability > 0.5 else "Healthy ✅"

        return jsonify({"prediction": result, "confidence": round(float(probability), 2)})

    except Exception as e:
        return jsonify({"error": f"Prediction Error: {str(e)}"})

@app.route("/ask", methods=["POST"])
def chatbot():
    user_input = request.json.get("question", "").strip().lower()
    responses = {
        "hi": "Hello! How can I assist you with Parkinson’s-related queries today?",
        "what is parkinson's ?": "Parkinson’s disease is a progressive nervous system disorder that affects movement.",
        "what are the symptoms ?": "Symptoms include tremors, stiffness, slowness of movement, and balance problems.",
        "is there a cure ?": "Currently, there is no cure for Parkinson's, but treatments can help manage symptoms.",
        "how is it diagnosed ?": "Diagnosis is based on medical history, symptoms, and neurological examinations.",
        "can it be prevented ?": "There is no guaranteed prevention, but a healthy lifestyle may lower the risk.",
        "what are the treatment options ?": "Treatment includes medication, therapy, and in some cases, surgical intervention like deep brain stimulation.",
        "what causes parkinson's ?": "The exact cause is unknown, but genetic and environmental factors are believed to play a role.",
        "who is at risk ?": "Risk factors include age, family history, exposure to toxins, and head injuries.",
        "does stress worsen parkinson’s ?": "Yes, stress can exacerbate symptoms, making tremors and movement issues more pronounced.",
        "can exercise help ?": "Yes, regular physical activity can improve mobility, balance, and overall quality of life.",
        "are there support groups ?": "Yes, there are many Parkinson’s support groups that provide resources and emotional support for patients and caregivers."
    }
    answer = responses.get(user_input, "I'm not sure about that. Try asking something else related to Parkinson’s.")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5500)
