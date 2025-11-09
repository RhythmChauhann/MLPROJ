#  Automated Machine Learning Pipeline — *Industry-Grade End-to-End Implementation*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![DagsHub](https://img.shields.io/badge/DagsHub-Integrated%20Tracking-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-lightblue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🧠 Project Overview

This repository implements a **complete, end-to-end Machine Learning pipeline**, following **real-world MLOps best practices**.  
Built using **Python**, this project leverages **DVC**, **MLflow**, and **DagsHub** to enable experiment tracking, data versioning, and full pipeline reproducibility.

The system trains multiple ML models on a dataset, evaluates them, and automatically identifies and stores the **best-performing model** — with all results logged for visualization and comparison.

---

## ⚙️ Key Features

✅ **End-to-End ML Pipeline** — from data ingestion to model deployment readiness  
✅ **MLflow Integration** — tracks all metrics, parameters, and artifacts  
✅ **DVC Integration** — ensures dataset and model reproducibility  
✅ **DagsHub Connectivity** — cloud-based tracking and visualization  
✅ **Multi-Model Evaluation** — trains and compares multiple ML models automatically  
✅ **Docker Support** — complete containerized environment  
✅ **Scalable Architecture** — modular and production-ready code design  

---

## 🧩 Tech Stack

| Component | Tool |
|------------|------|
| **Language** | Python 🐍 |
| **Version Control** | Git & GitHub |
| **Data Versioning** | DVC |
| **Experiment Tracking** | MLflow |
| **Remote Tracking & Visualization** | DagsHub |
| **Containerization** | Docker |
| **Logging** | Custom Python Logging |

---

## 📁 Project Structure

```plaintext
.
├── .dvc/                 # DVC metadata for data and model tracking
├── artifact/             # Stored artifacts such as trained models
├── catboost_info/        # Model training logs (CatBoost)
├── logs/                 # Custom logs for training and evaluation
├── notebook/             # Jupyter notebooks for experiments
├── src/mlproj/           # Core project source code
│   ├── __init__.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── evaluation.py
│   └── pipeline.py
├── .dvcignore            # Files ignored by DVC
├── .gitignore            # Files ignored by Git
├── Dockerfile            # Containerization setup
├── README.md             # Project documentation
├── app.py                # Main application entry point
├── requirements.txt      # Project dependencies
├── setup.py              # Setup configuration for packaging
└── template.py           # Initial project template

```
---
# 🧰 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/rhythmchauhann/MLPROJ.git
cd MLPROJ
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Initialize DVC
```bash
dvc init
dvc pull  # Download remote dataset if available
```

### 5️⃣ Run the main application
```bash
python app.py
```

---

## 📊 MLflow + DagsHub Tracking

Each model training run automatically logs:
- Parameters
- Metrics
- Model artifacts
- Run timestamps

All experiments can be visualized on your **DagsHub dashboard**:

🔗 [View MLflow Runs on DagsHub](https://dagshub.com/rhythmchauhann/MLPROJ)

---



### 🔍 Workflow Explanation
- **Data Loading** — Reads and validates the dataset.  
- **Preprocessing** — Handles cleaning, encoding, and transformations.  
- **Training** — Trains multiple ML models (Linear Regression, CatBoost, RandomForest, etc.).  
- **Evaluation** — Compares metrics and selects the top-performing model.  
- **Tracking** — Logs results and parameters with MLflow.  
- **Versioning** — Saves datasets and models with DVC for reproducibility.  

---

## 🧱 Docker Support

Run the entire project inside a Docker container for full reproducibility.

### 🐳 Build the Docker image
```bash
docker build -t mlproj .
```

### ▶️ Run the container
```bash
docker run -it mlproj
```

---

## 📈 Results

✅ Automatic multi-model evaluation  
✅ Best model saved and logged  
✅ Fully tracked experiments in MLflow  
✅ Versioned data and models with DVC  
✅ Dashboard integration via DagsHub  

---

## 💡 Future Improvements

- 🔁 Add automated CI/CD with GitHub Actions  
- ☁️ Integrate cloud storage (AWS S3 / GCP)  
- 🌐 Deploy REST API endpoints for inference  
- 📊 Add model monitoring and retraining pipeline  

---

## 🧑‍💻 Author

**Rhythm Chauhann**  
🎓 AI/ML Engineer | Data Science 

📍 **Connect:**  
- [GitHub](https://github.com/rhythmchauhann)  
- [DagsHub Project](https://dagshub.com/rhythmchauhann/MLPROJ)  

---

⭐ *If you like this project, don't forget to give it a star!* ⭐
