# Telco Customer Churn Prediction

A Machine Learning project to predict whether a customer is likely to churn based on their telecom usage and service features. The project includes an end-to-end ML pipeline, from EDA to deployment with a user-friendly web app using Streamlit.

---

## Objective

- Predict customer churn using features like contract type, monthly charges, internet service, etc.
- Build a clean, reusable ML pipeline using Scikit-learn
- Tune hyperparameters using GridSearchCV
- Deploy model as a Streamlit web application

---

## Dataset

- **Source**: [IBM Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Features include:
  - Demographics: Gender, SeniorCitizen, Partner, Dependents
  - Services: PhoneService, InternetService, StreamingTV, etc.
  - Charges: MonthlyCharges, TotalCharges
  - Target: `Churn` (Yes/No)

---

## Skills Gained

- Exploratory Data Analysis (EDA)
- Feature Engineering
- ML Pipelines using Scikit-learn
- Hyperparameter tuning with GridSearchCV
- Model evaluation (Accuracy, Precision, Recall, F1)
- Model serialization with Joblib
- Web app development with Streamlit

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Checked null values, data types, and imbalances
- Visualized:
  - Churn distribution
  - Numeric & categorical feature distributions
  - Correlation heatmap
  - Feature relationships with churn

### 2. Data Preprocessing

- Converted `TotalCharges` to numeric
- Removed `customerID`
- Encoded `Churn` to binary
- Used `ColumnTransformer` for:
  - Standard scaling numeric features
  - One-hot encoding categorical features

### 3. Model Building

- Split data using `train_test_split`
- Built pipelines for:
  - Logistic Regression
  - Random Forest
- Used `GridSearchCV` to find best hyperparameters

### 4. Evaluation

- Classification report: Accuracy, Precision, Recall, F1
- Chose the best model (based on cross-validation score)

### 5. Model Export

- Exported the best model using `joblib`:

```python

  joblib.dump(best_model, "telco_churn_model.joblib")
```

To run the app:

```python
    streamlit run streamlit_app.py # run the line on bash/terminal
```
