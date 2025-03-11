from flask import Flask, request, jsonify, render_template # type: ignore
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")


# Load the trained model
try:
    model = joblib.load("xgb_parkinsons_model.pkl")  # Update with correct model filename
    scaler = joblib.load("scaler.pkl")  # Load the scaler if required
except Exception as e:
    model = None
    print("Error loading model:", e)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Scale the input if required
        final_features = scaler.transform(final_features)

        # Make prediction
        if model:
            prediction = model.predict(final_features)
            return jsonify({"prediction": str(prediction[0])})
        else:
            return jsonify({"error": "Model not found!"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__=="__main__":
    app.run(debug=True)
