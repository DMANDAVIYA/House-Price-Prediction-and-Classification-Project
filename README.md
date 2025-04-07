# 🏠 House Price Prediction & Classification

## 🎯 Objective
The objective of this project is to:
- Predict house prices using **Linear Regression**
- Classify house prices into **Low**, **Medium**, and **High** using **K-Nearest Neighbors (KNN)**

---

## 📂 Dataset
- **File Name**: `house_Prediction_Data_Set.csv`
- **Features**: Multiple numerical features (e.g., area, bedrooms, etc.)
- **Target**: Final house price

### 🔧 Preprocessing Includes:
- Filling missing values using **median**
- Normalizing numerical features using **StandardScaler**
- Splitting dataset into **80% train** and **20% test**

---

## 🧠 ML Models Used

### 1. **Linear Regression** (for price prediction)
- Trained to predict continuous values of house prices.

### 2. **K-Nearest Neighbors Classifier** (for price category)
- Target categorized into:
  - **Low**: 0 to 15
  - **Medium**: 15 to 25
  - **High**: above 25

---

## 📊 Model Performance

### 🔹 Linear Regression:
| Metric                 | Score    |
|------------------------|----------|
| Mean Squared Error (MSE) | `24.29` |
| R-squared (R² Score)     | `0.67`  |  

### 🔹 KNN Classifier:
| Metric         | Score    |
|----------------|----------|
| Accuracy       | `0.86`   |  

📌 Also includes:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

---

## 📁 Project Structure

ML-Project/
├── data/                  # Contains raw dataset and preprocessed CSV files
│   ├── house_Prediction_Data_Set.csv
│   
│  
│   
│   
│
├── models/                # Trained and saved ML models
│   ├── linear_regression_model.pkl
│   └── final_knn_model.pkl
│
├── notebooks/             # Jupyter Notebooks for preprocessing, training, evaluation
│   └── PROJ.ipynb
│
├── docs/                  # Project documentation files (PDF or Markdown)
│   └── ML_PROJECT.pdf
│
├── src/                   # Python scripts for modular code (optional)
│   ├── preprocess.py
│   └── train_models.py
│
├── README.md              # Project overview and guide
├── requirements.txt       # Python dependencies
└── LICENSE                # Open-source license file (e.g., MIT)


## 💾 Saved Models
- `models/linear_regression_model.pkl`
- `models/final_knn_model.pkl`

Load with:
```python
import joblib
lr_model = joblib.load('models/linear_regression_model.pkl')
knn_model = joblib.load('models/final_knn_model.pkl')
