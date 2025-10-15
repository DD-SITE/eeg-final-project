from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os
import numpy as np

# -----------------------------
# Load preprocessing & model(s)
# -----------------------------
preprocessor = joblib.load("preprocessor.pkl")   # VarianceThreshold + PCA
ensemble_models = joblib.load("ensemble.pkl")   # Top trained classifiers per batch

# HTML template
HTML_PAGE = """
<!doctype html>
<title>EEG Classifier</title>
<h1>Upload EEG CSV file</h1>
<form action="/predict" method=post enctype=multipart/form-data>
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
{% if prediction %}
  <h2>Prediction Result:</h2>
  <pre>{{ prediction }}</pre>
{% endif %}
"""

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    # Read uploaded CSV
    df = pd.read_csv(file)

    # Transform features using saved preprocessing pipeline
    X_input = preprocessor.transform(df)

    # Predict using all top models in ensemble
    predictions = {}
    for batch_idx, batch_models in enumerate(ensemble_models):
        for name, clf, _ in batch_models:  # unpack (name, model, test_acc)
            preds = clf.predict(X_input)
            predictions[f"Batch{batch_idx+1}_{name}"] = preds

    # Convert to DataFrame for display
    result_df = pd.DataFrame(predictions)

    return render_template_string(HTML_PAGE, prediction=result_df.to_string(index=False))

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
