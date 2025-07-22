# Fraud Detection for E-Commerce and Bank Transactions

## Project Overview
This project aims to improve the detection of fraud cases for e-commerce and bank credit transactions using advanced data analysis and machine learning techniques. The workflow covers data cleaning, feature engineering, handling class imbalance, model building, evaluation, and comparison.

---

## Task 1: Data Analysis and Preprocessing

### Steps Completed
- **Data Loading:** Imported `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv`.
- **Data Cleaning:**
  - Removed duplicates.
  - Converted date columns to datetime.
  - Checked and handled missing values.
- **Feature Engineering:**
  - Created transaction frequency features per user and device.
  - Extracted time-based features: hour of day, day of week, and time since signup.
  - Mapped IP addresses to countries.
- **Exploratory Data Analysis (EDA):**
  - Visualized distributions of key features (e.g., purchase value, age, class).
  - Explored relationships between features and fraud occurrence.
- **Class Imbalance Analysis:**
  - Analyzed class distribution in both datasets.
  - Planned to use SMOTE for oversampling during model training.

---

## Task 2: Model Building and Training

### Steps Completed
- **Data Preparation:**
  - Separated features and target variables for both datasets.
  - Performed train-test split with stratification.
- **Preprocessing:**
  - Encoded categorical features using OneHotEncoder.
  - Scaled numerical features using StandardScaler.
- **Handling Class Imbalance:**
  - Applied SMOTE oversampling to the training data.
- **Model Training:**
  - Trained Logistic Regression and Random Forest models on both datasets.
- **Evaluation:**
  - Evaluated models using F1-score, AUC-PR, confusion matrix, and classification report.
  - Compared model performances to select the best approach for each dataset.

---

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the data inspection and preprocessing script:
   ```bash
   python data_inspection.py
   ```
3. Run the model building and evaluation script:
   ```bash
   python model_building.py
   ```

---

## Project Structure
- `data_inspection.py` — Data loading, cleaning, EDA, and feature engineering.
- `model_building.py` — Preprocessing, balancing, model training, and evaluation.
- `Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv` — Datasets.

---

## Next Steps
- Model explainability with SHAP (Task 3).
- Final report and documentation.

---

## Authors
- Data Scientist: [Your Name]
- Company: Adey Innovations Inc.
# fraud-detaction-for-e-commerce-and-bank-transactios