from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os

# Load your trained model
model = joblib.load("model.pkl")

# HTML template with CSS for better UI
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
.pred-0 { color: red; font-weight: bold; }
.pred-1 { color: green; font-weight: bold; }
.summary { font-weight: bold; margin-bottom: 15px; text-align: center; }
</style>
</head>
<body>
<div class="container">
<h1>EEG Classifier</h1>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <input type="submit" value="Upload & Predict">
</form>

{% if summary %}
<div class="summary">{{ summary }}</div>
{% endif %}

{% if prediction %}
<div class="table-container">
{{ prediction|safe }}
</div>
{% endif %}
</div>
</body>
</html>
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

    df = pd.read_csv(file)

    expected_features = model.n_features_in_
    current_features = df.shape[1]

    # Add missing columns all at once to prevent fragmentation
    if current_features < expected_features:
        missing_cols = pd.DataFrame(0, index=df.index, columns=[f"missing_{i}" for i in range(expected_features - current_features)])
        df = pd.concat([df, missing_cols], axis=1)
    elif current_features > expected_features:
        df = df.iloc[:, :expected_features]

    # Predict directly on DataFrame to retain feature names
    preds = model.predict(df)

    result_df = pd.DataFrame({
        "Row": range(1, len(preds) + 1),
        "Predicted_Class": preds.astype(int)
    })

    count_0 = (preds == 0).sum()
    count_1 = (preds == 1).sum()
    summary_text = f"Predicted 1: {count_1} samples, Predicted 0: {count_0} samples"

    # Style results with color for classes and render as HTML table
    result_html = result_df.style.map(
        lambda v: 'color: green; font-weight: bold;' if v == 1 else 'color: red; font-weight: bold;' if v == 0 else '',
        subset=['Predicted_Class']
    ).to_html()

    return render_template_string(HTML_PAGE, prediction=result_html, summary=summary_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
