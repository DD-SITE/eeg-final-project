from flask import Flask, render_template, request
import numpy as np
import joblib
import mne
from scipy.signal import welch

app = Flask(__name__)

# Load models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

try:
    pca = joblib.load("pca.pkl")
    use_pca = True
except:
    use_pca = False


def bandpower(signal, sf=500, band=(8,13)):
    f, Pxx = welch(signal, sf, nperseg=256)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.sum(Pxx[idx])


def extract_features(segment):
    features = []
    for ch in range(segment.shape[1]):
        sig = segment[:, ch]
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(bandpower(sig, band=(8,13)))
        features.append(bandpower(sig, band=(13,30)))
    return features


def process_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    eeg_channels = [
        'EEG Fp1','EEG Fp2','EEG F3','EEG F4','EEG F7','EEG F8',
        'EEG T3','EEG T4','EEG C3','EEG C4','EEG T5','EEG T6',
        'EEG P3','EEG P4','EEG O1','EEG O2','EEG Fz','EEG Cz','EEG Pz'
    ]

    raw.pick(eeg_channels)
    raw.filter(0.5, 45)

    data = raw.get_data().T

    # Take only first segment (fast for free tier)
    segment = data[:500]

    features = extract_features(segment)

    X = np.array(features).reshape(1, -1)
    X = scaler.transform(X)

    if use_pca:
        X = pca.transform(X)

    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob > 0.30 else 0

    return pred, prob


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None

    if request.method == "POST":
        file = request.files["file"]
        path = "temp.edf"
        file.save(path)

        pred, prob = process_edf(path)

        result = "Good Performance" if pred == 1 else "Bad Performance"

    return render_template("index.html", result=result, prob=prob)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
