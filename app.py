from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")   # ✅ IMPORTANT for Render
import matplotlib.pyplot as plt
import io
import base64

# -----------------------
# LOAD MODEL + FEATURES
# -----------------------

model = joblib.load("model.pkl")

# Load feature names (IMPORTANT)
try:
    feature_names = joblib.load("features.pkl")
except:
    feature_names = None   # fallback

# -----------------------
# HTML TEMPLATE
# -----------------------

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EEG Classifier</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; }
.container { max-width: 1100px; margin: 40px auto; background: white; padding: 30px; border-radius: 10px; }
h1, h2 { text-align: center; }
form { text-align: center; margin-bottom: 20px; }
input[type=submit] { padding: 10px 20px; background: green; color: white; border: none; border-radius: 5px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 8px; border: 1px solid #ddd; text-align: center; }
.summary { text-align: center; font-weight: bold; margin-bottom: 15px; }
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
{{ prediction|safe }}
{% endif %}

{% if feature_importance %}
<h2>Top Feature Importance</h2>
{{ feature_importance|safe }}
{% endif %}

{% if feature_plot %}
<h2>Feature Importance Graph</h2>
<img src="data:image/png;base64,{{ feature_plot }}">
{% endif %}

</div>
</body>
</html>
"""

app = Flask(__name__)

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def confidence_label(p):
    if p is None:
        return "N/A"
    elif p > 0.8:
        return "High"
    elif p > 0.6:
        return "Medium"
    else:
        return "Low"


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_") and feature_names:
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
    return None


def plot_feature_importance(df):
    plt.figure(figsize=(8, 5))
    plt.barh(df["Feature"], df["Importance"])
    plt.gca().invert_yaxis()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# -----------------------
# ROUTES
# -----------------------

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/health")
def health():
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file:
        return "No file uploaded", 400

    # ✅ Validate file
    if not file.filename.endswith(".csv"):
        return "Upload a CSV file only", 400

    try:
        df = pd.read_csv(file)
    except:
        return "Invalid CSV file", 400

    # ✅ Limit size (performance)
    df = df.head(500)

    # ✅ FIX: Align features properly
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    # -----------------------
    # PREDICTIONS
    # -----------------------

    preds = model.predict(df)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        prob_class1 = probs[:, 1]
    else:
        prob_class1 = [None] * len(preds)

    result_df = pd.DataFrame({
        "Row": range(1, len(preds) + 1),
        "Prediction": preds,
        "Confidence": prob_class1
    })

    result_df["Confidence_Level"] = result_df["Confidence"].apply(confidence_label)

    # Limit rows in UI
    result_df = result_df.head(100)

    # -----------------------
    # SUMMARY
    # -----------------------

    count1 = (preds == 1).sum()
    count0 = (preds == 0).sum()

    valid_probs = [p for p in prob_class1 if p is not None]

    if valid_probs:
        avg_conf = sum(valid_probs) / len(valid_probs)
    else:
        avg_conf = 0

    summary = f"Class1: {count1} | Class0: {count0} | Avg Confidence: {avg_conf:.2f}"

    prediction_html = result_df.to_html(index=False)

    # -----------------------
    # FEATURE IMPORTANCE
    # -----------------------

    fi_df = get_feature_importance(model, feature_names)

    if fi_df is not None:
        top = fi_df.head(10)
        feature_html = top.to_html(index=False)
        feature_plot = plot_feature_importance(top)
    else:
        feature_html = None
        feature_plot = None

    return render_template_string(
        HTML_PAGE,
        prediction=prediction_html,
        summary=summary,
        feature_importance=feature_html,
        feature_plot=feature_plot
    )


# -----------------------
# MAIN
# -----------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
