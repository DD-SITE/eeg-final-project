from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # must include feature extraction steps

# -----------------------------
# HTML template with styling
# -----------------------------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EEG Classifier</title>
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
.container { max-width: 900px; margin: 40px auto; background: #fff; padding: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-radius: 10px; }
h1 { text-align: center; color: #333; }
form { text-align: center; margin-bottom: 25px; }
input[type=file] { padding: 8px; }
input[type=submit] { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
input[type=submit]:hover { background: #45a049; }
.table-container { overflow-x:auto; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
th { background-color: #4CAF50; color: white; }
tr:nth-child(even){background-color: #f9f9f9;}
tr:hover {background-color: #f1f1f1;}
</style>
</head>
<body>
<div class="container">
<h1>EEG Classification</h1>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <input type="submit" value="Upload & Predict">
</form>

{% if prediction %}
<div class="table-container">
<h2>Prediction Results:</h2>
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

    # Read uploaded CSV
    raw_df = pd.read_csv(file)

    # Preprocess features
    processed_features = preprocessor.transform(raw_df)

    # Predict
    preds = model.predict(processed_features)

    # Convert to DataFrame for nice table display
    result_df = pd.DataFrame(preds, columns=["Predicted_Class"])
    result_html = result_df.to_html(index=True, border=0)

    return render_template_string(HTML_PAGE, prediction=result_html)

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
