import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")

# HTML template
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EEG Classifier</title>
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f0f2f5;
    margin: 0;
    padding: 0;
}
.container {
    max-width: 1000px;
    margin: 40px auto;
    background: #fff;
    padding: 30px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    border-radius: 10px;
}
h1 {
    text-align: center;
    color: #333;
}
form {
    text-align: center;
    margin-bottom: 25px;
}
input[type=file] {
    padding: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    text-align: center;
}
tr:hover {
    background-color: #f1f1f1;
}
.pred-0 {
    color: red;
    font-weight: bold;
}
.pred-1 {
    color: green;
    font-weight: bold;
}
.high {
    color: green;
    font-weight: bold;
}
.medium {
    color: orange;
    font-weight: bold;
}
.low {
    color: red;
    font-weight: bold;
}
.summary {
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
}
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
<div class="summary">
    {{ summary }}
</div>
{% endif %}

{% if prediction %}
<div>
    {{ prediction|safe }}
</div>
{% endif %}

</div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


def confidence_label(p):
    if p is None:
        return "N/A"
    elif p > 0.8:
        return "High"
    elif p > 0.6:
        return "Medium"
    else:
        return "Low"


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file:
        return render_template_string(HTML_PAGE, prediction="No file uploaded")

    df = pd.read_csv(file)

    # Match feature count
    expected_features = model.n_features_in_
    current_features = df.shape[1]

    if current_features < expected_features:
        missing_cols = pd.DataFrame(
            0,
            index=df.index,
            columns=[f"missing_{i}" for i in range(expected_features - current_features)]
        )
        df = pd.concat([df, missing_cols], axis=1)

    elif current_features > expected_features:
        df = df.iloc[:, :expected_features]

    # Predictions
    preds = model.predict(df)

    # Probabilities (if supported)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        prob_class1 = probs[:, 1]
    else:
        prob_class1 = [None] * len(preds)

    # Create result dataframe
    result_df = pd.DataFrame({
        "Row": range(1, len(preds) + 1),
        "Predicted_Class": preds.astype(int),
        "Confidence": prob_class1
    })

    # Add confidence level
    result_df["Confidence_Level"] = result_df["Confidence"].apply(confidence_label)

    # Summary stats
    count_0 = (preds == 0).sum()
    count_1 = (preds == 1).sum()

    valid_probs = [p for p in prob_class1 if p is not None]
    avg_conf = sum(valid_probs) / len(valid_probs) if valid_probs else 0

    summary_text = f"""
    Predicted 1: {count_1} samples |
    Predicted 0: {count_0} samples |
    Avg Confidence: {avg_conf:.2f}
    """

    # Styling functions
    def style_prediction(val):
        return 'color: green; font-weight: bold;' if val == 1 else 'color: red; font-weight: bold;'

    def style_confidence(val):
        if val == "High":
            return 'color: green; font-weight: bold;'
        elif val == "Medium":
            return 'color: orange; font-weight: bold;'
        elif val == "Low":
            return 'color: red; font-weight: bold;'
        return ''

    styled_table = (
        result_df.style
        .map(style_prediction, subset=["Predicted_Class"])
        .map(style_confidence, subset=["Confidence_Level"])
        .format({"Confidence": "{:.2f}"})
        .to_html()
    )

    return render_template_string(
        HTML_PAGE,
        prediction=styled_table,
        summary=summary_text
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
