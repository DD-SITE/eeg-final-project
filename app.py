from flask import Flask, request, render_template_string
import pandas as pd
import joblib

# Load your trained pipeline
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
    file = request.files["file"]
    if not file:
        return "No file uploaded", 400

    # Read the uploaded CSV
    df = pd.read_csv(file)

    # Predict using the trained model
    preds = model.predict(df)

    # Convert results to DataFrame for clarity
    result_df = pd.DataFrame(preds, columns=["Predicted_Class"])

    return render_template_string(HTML_PAGE, prediction=result_df.to_string(index=False))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
