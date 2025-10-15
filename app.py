from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Load ensemble models only
# -----------------------------
ensemble_models = joblib.load("ensemble.pkl")  # top_models_per_batch

# -----------------------------
# HTML template
# -----------------------------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EEG Classifier</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; }
h1 { color: #333; }
form { margin-bottom: 20px; }
input[type=file] { padding: 5px; }
input[type=submit] { padding: 8px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
input[type=submit]:hover { background: #45a049; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
th { background-color: #4CAF50; color: white; }
tr:nth-child(even){background-color: #f2f2f2;}
pre { white-space: pre-wrap; word-wrap: break-word; }
.container { max-width: 1000px; margin: auto; }
</style>
</head>
<body>
<div class="container">
<h1>EEG Classifier</h1>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <input type="submit" value="Upload & Predict">
</form>
{% if prediction %}
<h2>Prediction Result:</h2>
<div style="overflow-x:auto;">
{{ prediction|safe }}
</div>
{% endif %}
</div>
</body>
</html>
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

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error reading CSV: {e}", 400

    # -------------------------
    # Pad CSV to match ensemble input (2000 features)
    # -------------------------
    expected_features = 2000
    current_features = df.shape[1]

    if current_features < expected_features:
        for i in range(expected_features - current_features):
            df[f"missing_{i}"] = 0
    elif current_features > expected_features:
        df = df.iloc[:, :expected_features]

    X_input = df.to_numpy()

    # -------------------------
    # Predict using ensemble
    # -------------------------
    predictions = {}
    for batch_idx, batch_models in enumerate(ensemble_models):
        for name, clf, _ in batch_models:
            try:
                preds = clf.predict(X_input)
                predictions[f"Batch{batch_idx+1}_{name}"] = preds
            except Exception as e:
                predictions[f"Batch{batch_idx+1}_{name}"] = f"Error: {e}"

    # -------------------------
    # Convert to nice HTML table
    # -------------------------
    result_df = pd.DataFrame(predictions)
    result_html = result_df.to_html(index=True, classes="table table-striped", border=0)

    return render_template_string(HTML_PAGE, prediction=result_html)

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
