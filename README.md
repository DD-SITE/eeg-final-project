# 🧠 EEG Signal Classification Web App  
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/) 
[![Flask](https://img.shields.io/badge/Flask-Framework-black.svg)](https://flask.palletsprojects.com/) 
[![Machine Learning](https://img.shields.io/badge/Model-Ensemble-success.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue.svg)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-green.svg)](https://render.com)


A web-based machine learning application that classifies **EEG (Electroencephalogram) signals** into different mental states using an **ensemble of ML models** trained on EEG features.  
Built with **Python, Flask, and Scikit-Learn**, containerized with **Docker**, and deployed seamlessly on **Render** 🚀  

🔗 **Live App:** [https://eeg-final-project.onrender.com](https://eeg-final-project.onrender.com)

---

## 🧩 Features
- 📂 Upload EEG data (CSV format)
- ⚙️ Automatically preprocesses and feeds data to a trained ensemble model
- 🧠 Predicts the EEG signal class (mental state)
- 📊 Displays prediction results in an interactive web interface
- 🌐 Fully containerized and deployed on Render using Docker

---

## 🧠 Tech Stack
| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | HTML, CSS (Flask templates) |
| **Backend** | Python, Flask |
| **Machine Learning** | Scikit-Learn, NumPy, Pandas |
| **Model** | Trained Ensemble (`model.pkl`) |
| **Containerization** | Docker |
| **Deployment** | Render Cloud Platform |

---

## ⚙️ Installation (Local Setup)

To run this project locally:

```bash
# Clone the repository
git clone https://github.com/<your-username>/EEG_Final_Project.git
cd EEG_Final_Project

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
#Then open your browser and go to:
http://localhost:8080
```


---

## 🧪 Model Details
The notebook explores and trains multiple models:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**
- **CatBoost**

An **ensemble model** was created to combine the strengths of individual models, achieving higher accuracy and stability.  
The final trained model was saved as **`model.pkl`**, which is used in the deployed web app for real-time inference.

---

## 🚀 Deployment
This project is fully containerized using **Docker** and deployed on **Render**.

**Dockerfile**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]
```
**Render Configuration**

Environment: Docker

Port: 8080

Region: Singapore (for Indian users, best latency)
---
## 📦 Dataset  

This project uses the **Complete EEG Dataset** available publicly on Kaggle:  
🔗 [Complete EEG Dataset – Kaggle](https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset)  

The dataset contains EEG signal data from multiple subjects, captured under different mental states and experimental conditions. Each record consists of a series of EEG channel readings and a label representing the associated cognitive state.  

> ⚠️ **Note:** The full dataset is **not included** in this repository due to size constraints.  
> To train or retrain the model, download it from the Kaggle link above and place the required `.csv` file(s) in the project directory.  

For demonstration purposes, a lightweight `sample_data.csv` file (containing only a few rows) is provided.  
You can use this sample to:  
- Test the live web app upload functionality  
- Quickly verify the pipeline on your local machine  
- Understand the expected input format  

