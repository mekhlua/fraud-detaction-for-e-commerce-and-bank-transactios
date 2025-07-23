import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score
import shap

fraud = pd.read_csv('Fraud_Data.csv')
credit = pd.read_csv('creditcard.csv')

# Remove duplicates
fraud = fraud.drop_duplicates()
credit = credit.drop_duplicates()

# Convert date columns to datetime
fraud['signup_time'] = pd.to_datetime(fraud['signup_time'])
fraud['purchase_time'] = pd.to_datetime(fraud['purchase_time'])

# Transaction frequency per user
fraud['user_txn_count'] = fraud.groupby('user_id')['user_id'].transform('count')
# Transaction frequency per device
fraud['device_txn_count'] = fraud.groupby('device_id')['device_id'].transform('count')
# Time-based features
fraud['hour_of_day'] = fraud['purchase_time'].dt.hour
fraud['day_of_week'] = fraud['purchase_time'].dt.dayofweek
fraud['time_since_signup'] = (fraud['purchase_time'] - fraud['signup_time']).dt.total_seconds() / 3600  # in hours

# For e-commerce fraud data
X_fraud = fraud.drop(columns=['class'])
y_fraud = fraud['class']

# Identify categorical and numerical columns
cat_cols = ['source', 'browser', 'sex', 'device_id']
num_cols = X_fraud.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col not in ['user_id', 'device_id', 'ip_int']]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Split data
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

# Fit transform on train, transform on test
Xf_train_prep = preprocessor.fit_transform(Xf_train)
Xf_test_prep = preprocessor.transform(Xf_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
Xf_train_bal, yf_train_bal = smote.fit_resample(Xf_train_prep, yf_train)

# For credit card data

# For credit card data
X_credit = credit.drop(columns=['Class'])
y_credit = credit['Class']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit)

# Identify numerical columns (all except 'Time' and 'Amount' are anonymized PCA features)
num_cols_credit = X_credit.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessing pipeline for credit data (scale all features)
preprocessor_credit = ColumnTransformer([
    ('num', StandardScaler(), num_cols_credit)
])

# Fit transform on train, transform on test
Xc_train_prep = preprocessor_credit.fit_transform(Xc_train)
Xc_test_prep = preprocessor_credit.transform(Xc_test)

# Handle class imbalance with SMOTE
smote_credit = SMOTE(random_state=42)
Xc_train_bal, yc_train_bal = smote_credit.fit_resample(Xc_train_prep, yc_train)

# Logistic Regression for credit data
lr_credit = LogisticRegression(max_iter=1000, random_state=42)
lr_credit.fit(Xc_train_bal, yc_train_bal)
y_pred_lr_credit = lr_credit.predict(Xc_test_prep)
print("\nCredit Data - Logistic Regression Results:")
print(classification_report(yc_test, y_pred_lr_credit))
print("Confusion Matrix:\n", confusion_matrix(yc_test, y_pred_lr_credit))
print("F1 Score:", f1_score(yc_test, y_pred_lr_credit))
print("AUC-PR:", average_precision_score(yc_test, lr_credit.predict_proba(Xc_test_prep)[:,1]))

# Random Forest for credit data
rf_credit = RandomForestClassifier(n_estimators=100, random_state=42)
rf_credit.fit(Xc_train_bal, yc_train_bal)
y_pred_rf_credit = rf_credit.predict(Xc_test_prep)
print("\nCredit Data - Random Forest Results:")
print(classification_report(yc_test, y_pred_rf_credit))
print("Confusion Matrix:\n", confusion_matrix(yc_test, y_pred_rf_credit))
print("F1 Score:", f1_score(yc_test, y_pred_rf_credit))
print("AUC-PR:", average_precision_score(yc_test, rf_credit.predict_proba(Xc_test_prep)[:,1]))
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(Xf_train_bal, yf_train_bal)
y_pred_lr = lr.predict(Xf_test_prep)
print("Logistic Regression Results:")
print(classification_report(yf_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(yf_test, y_pred_lr))
print("F1 Score:", f1_score(yf_test, y_pred_lr))
print("AUC-PR:", average_precision_score(yf_test, lr.predict_proba(Xf_test_prep)[:,1]))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(Xf_train_bal, yf_train_bal)
y_pred_rf = rf.predict(Xf_test_prep)
print("Random Forest Results:")
print(classification_report(yf_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(yf_test, y_pred_rf))
print("F1 Score:", f1_score(yf_test, y_pred_rf))
print("AUC-PR:", average_precision_score(yf_test, rf.predict_proba(Xf_test_prep)[:,1]))

# For e-commerce fraud Random Forest
explainer_fraud = shap.TreeExplainer(rf)
shap_values_fraud = explainer_fraud.shap_values(Xf_test_prep.toarray().astype(float))

print("E-commerce Fraud Model SHAP Summary Plot:")
shap.summary_plot(shap_values_fraud[1], Xf_test_prep, feature_names=preprocessor.get_feature_names_out())

# For credit card Random Forest
explainer_credit = shap.TreeExplainer(rf_credit)
shap_values_credit = explainer_credit.shap_values(Xc_test_prep.toarray().astype(float)) 

print("Credit Card Model SHAP Summary Plot:")
shap.summary_plot(shap_values_credit[1], Xc_test_prep, feature_names=preprocessor_credit.get_feature_names_out())

# Local explanation for a single prediction (e.g., first test sample)
shap.initjs()
print("E-commerce Fraud Model SHAP Force Plot (first test sample):")
shap.force_plot(
    explainer_fraud.expected_value[1],
    shap_values_fraud[1][0],
    Xf_test_prep[0],
    feature_names=preprocessor.get_feature_names_out(),
    matplotlib=True
)

print("Credit Card Model SHAP Force Plot (first test sample):")
shap.force_plot(
    explainer_credit.expected_value[1],
    shap_values_credit[1][0],
    Xc_test_prep[0],
    feature_names=preprocessor_credit.get_feature_names_out(),
    matplotlib=True
)
