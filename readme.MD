# 🔮 Telco Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4223?style=for-the-badge&logo=xgboost&logoColor=white)

## 📖 Project Overview
This project is an end-to-end **Machine Learning Engineering (MLE)** solution designed to predict customer churn in the telecommunications sector. 

Unlike standard notebooks, this project demonstrates a **production-ready architecture**:
1.  **Training:** An optimized XGBoost pipeline with custom feature engineering.
2.  **Deployment:** A containerized FastAPI microservice hosted on **AWS EC2**.
3.  **Interface:** A user-friendly **Streamlit** dashboard for real-time risk assessment.

> **🔴 Live Demo:** [INSERT_YOUR_STREAMLIT_APP_LINK_HERE]

---

## 🏗️ System Architecture
The application follows a decoupled microservices pattern:

```mermaid
graph LR
    A[User (Streamlit UI)] -- "JSON Request" --> B[AWS Cloud (EC2)]
    subgraph "Docker Container"
    B --> C[FastAPI Server]
    C -- "Features" --> D[XGBoost Model]
    D -- "Prediction (Risk %)" --> C
    end
    C -- "Response" --> A
🛠️ Tech Stack
Frontend & User Interface

Streamlit: Serves the interactive dashboard. Allows non-technical stakeholders to test the model in real-time.

Backend & API

FastAPI: Chosen for its asynchronous capabilities and automatic Swagger UI documentation.
Pydantic: strict data validation ensures the model never receives malformed input.
Docker: Containerizes the application environment, ensuring identical performance from development to cloud.

Machine Learning

XGBoost Classifier: The core model, optimized via GridSearch for high recall (minimizing missed churners).
Scikit-Learn Pipelines: Handles preprocessing (OneHotEncoding, Scaling) and custom feature engineering logic.

Cloud Infrastructure

AWS EC2 (Ubuntu Linux): Hosts the Docker container, exposing the API via port 8000.

📂 Repository Structure
text├── app/                  # FastAPI Application Logic
│   └── api.py            # Main API entry point
├── churn_ui/             # Frontend Dashboard
│   ├── churn_app.py      # Streamlit App
│   └── requirements.txt  # Frontend specific dependencies
├── data/                 # Dataset storage (Raw CSVs)
├── models/               # Serialized ML Models
│   └── model_xgb.pkl     # Production XGBoost Model
├── src/                  # Source Code
│   ├── features.py       # Custom Feature Engineering Classes
│   ├── preprocessing.py  # Data cleaning & Splitting
│   └── train.py          # Training Pipeline
├── Dockerfile            # Blueprint for building the API image
├── requirements.txt      # Backend Python dependencies
└── README.md             # Project Documentation
⚡ Quick Start Guide

Clone the RepositoryBashgit clone https://github.com/YOUR_USERNAME/churn-prediction-system.git
cd churn-prediction-system
Run the Backend (Docker)
Ensure you have Docker installed.Bash# Build the image
docker build -t churn-api .

# Run container (Maps port 8000)
docker run -p 8000:8000 churn-apiAPI is now live at http://localhost:8000
Run the Frontend (Local)
Open a new terminal:Bashcd churn_ui
pip install -r requirements.txt
streamlit run churn_app.py

🔌 API Reference
Once the container is running, access the auto-generated Swagger documentation:
URL: http://localhost:8000/docs
Sample Request Body:
JSON{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
👨‍💻 Author
Aseem Garg

LinkedIn
GitHub

1.8stech stack and everything everything in one code blockMarkdown# 🔮 Telco Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4223?style=for-the-badge&logo=xgboost&logoColor=white)

## 📖 Project Overview
This project is an end-to-end **Machine Learning Engineering (MLE)** solution designed to predict customer churn in the telecommunications sector. 

Unlike standard notebooks, this project demonstrates a **production-ready architecture**:
1.  **Training:** An optimized XGBoost pipeline with custom feature engineering.
2.  **Deployment:** A containerized FastAPI microservice hosted on **AWS EC2**.
3.  **Interface:** A user-friendly **Streamlit** dashboard for real-time risk assessment.

> **🔴 Live Demo:** [INSERT_YOUR_STREAMLIT_APP_LINK_HERE]

---

## 🏗️ System Architecture
The application follows a decoupled microservices pattern:

```mermaid
graph LR
    A[User (Streamlit UI)] -- "JSON Request" --> B[AWS Cloud (EC2)]
    subgraph "Docker Container"
    B --> C[FastAPI Server]
    C -- "Features" --> D[XGBoost Model]
    D -- "Prediction (Risk %)" --> C
    end
    C -- "Response" --> A
🛠️ Tech Stack
Frontend & User Interface

Streamlit: Serves the interactive dashboard. Allows non-technical stakeholders to test the model in real-time.

Backend & API

FastAPI: Chosen for its asynchronous capabilities and automatic Swagger UI documentation.
Pydantic: strict data validation ensures the model never receives malformed input.
Docker: Containerizes the application environment, ensuring identical performance from development to cloud.

Machine Learning

XGBoost Classifier: The core model, optimized via GridSearch for high recall (minimizing missed churners).
Scikit-Learn Pipelines: Handles preprocessing (OneHotEncoding, Scaling) and custom feature engineering logic.

Cloud Infrastructure

AWS EC2 (Ubuntu Linux): Hosts the Docker container, exposing the API via port 8000.

📂 Repository Structure
text├── app/              # FastAPI Application Logic
│   └── api.py            # Main API entry point
├── churn_ui/             # Frontend Dashboard
│   ├── churn_app.py      # Streamlit App
│   └── requirements.txt  # Frontend specific dependencies
├── data/                 # Dataset storage (Raw CSVs)
├── models/               # Serialized ML Models
│   └── model_xgb.pkl     # Production XGBoost Model
├── src/                  # Source Code
│   ├── features.py       # Custom Feature Engineering Classes
│   ├── preprocessing.py  # Data cleaning & Splitting
│   └── train.py          # Training Pipeline
├── Dockerfile            # Blueprint for building the API image
├── requirements.txt      # Backend Python dependencies
└── README.md             # Project Documentation
⚡ Quick Start Guide

Clone the RepositoryBashgit clone https://github.com/YOUR_USERNAME/churn-prediction-system.git
cd churn-prediction-system
Run the Backend (Docker)
Ensure you have Docker installed.Bash# Build the image
docker build -t churn-api .

# Run container (Maps port 8000)
docker run -p 8000:8000 churn-apiAPI is now live at http://localhost:8000
Run the Frontend (Local)
Open a new terminal:Bashcd churn_ui
pip install -r requirements.txt
streamlit run churn_app.py

🔌 API Reference
Once the container is running, access the auto-generated Swagger documentation:
URL: http://localhost:8000/docs
Sample Request Body:
JSON{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
👨‍💻 Author
Aseem Garg