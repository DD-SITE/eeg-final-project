from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

# Load your trained pipeline/model
model = joblib.load("model.pkl")

# Basic HTML page
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

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    # Read the uploaded CSV
    df = pd.read_csv(file)

    # Ensure df has the correct number of features
    expected_features = model.n_features_in_
    current_features = df.shape[1]

    if current_features < expected_features:
        # Pad missing columns with zeros
        for i in range(expected_features - current_features):
            df[f"missing_{i}"] = 0
    elif current_features > expected_features:
        # Drop extra columns
        df = df.iloc[:, :expected_features]

    # Convert to NumPy array to avoid feature name warnings
    X_input = df.to_numpy()

    # Make predictions
    preds = model.predict(X_input)

    # Convert results to DataFrame for display
    result_df = pd.DataFrame(preds, columns=["Predicted_Class"])

    return render_template_string(HTML_PAGE, prediction=result_df.to_string(index=False))

if __name__ == "__main__":
    # Use PORT environment variable for Render
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
