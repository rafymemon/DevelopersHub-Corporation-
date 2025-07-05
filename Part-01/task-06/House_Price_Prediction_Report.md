
# House Price Prediction Project

This project involves building a machine learning model to predict house prices using features like square footage, number of bedrooms, and location.

---

## Objective

- Predict house prices using various property features.
- Perform preprocessing including handling numeric and categorical data.
- Train a regression model (Gradient Boosting Regressor).
- Visualize predicted vs actual prices.
- Evaluate the model using MAE and RMSE.

---

## Dataset Used

- Source: Kaggle - House Price Prediction Dataset
- **Target column:** `Price`
- **Features:** `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`, `Location`, `Condition`, `Garage`

---

## Libraries used throughout the task

```python
# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

---

## Steps Followed

### Step 1: Data Loading & Splitting

```python
# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['Price']),
    data['Price'],
    test_size=0.2,
    random_state=42
)
```

---

### Step 2: Preprocessing

```python
# Define feature types
numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
categorical_features = ['Location', 'Condition', 'Garage']

# Create ColumnTransformer for both types
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
])
```

---

### Step 3: Model Pipeline

```python
# Combine preprocessing and model into pipeline
pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])
```

---

### Step 4: Train the Model

```python
# Fit the pipeline
pipe.fit(X_train, y_train)
```

---

### Step 5: Evaluate the Model

```python
# Predict on test data
y_pred = pipe.predict(X_test)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
```

---

### Step 6: Visualization

```python
# Plot Actual vs Predicted Prices using scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()
```

```python
# Plotting using Lineplot

comparison_df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': y_pred
}).sort_values(by='Actual').reset_index(drop=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Actual'], label='Actual Price', linewidth=2)
plt.plot(comparison_df['Predicted'], label='Predicted Price', linewidth=2, linestyle='--')
plt.title('Comparison of Actual and Predicted House Prices')
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

### Summarizing the task

- The model was trained using a clean and professional pipeline structure.
- Both numeric and categorical features were preprocessed properly.
- The model showed reasonable prediction accuracy with MAE and RMSE.
- Visualization confirmed that predictions align closely with actual prices.

---

## Tools that has been used throughout the task

- **Python**
- **scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

---

## Author

Rafy Mustafa — Software Engineering Student at MUET Jasmhoro
Intern at — Developer’s Hub  
