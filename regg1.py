import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Load Dataset
# Replace 'housing_data.csv' with your dataset path
data = pd.read_csv('C:/Users/user5/Downloads/Housing (2).csv')

# Data Preprocessing
print(data.head())
print(data.info())

# Handle Missing Values
if data.isnull().sum().any():
    data.fillna(data.mean(numeric_only=True), inplace=True)  # Use mean only for numeric columns

# Feature Selection
if 'price' not in data.columns:
    raise KeyError("The target variable 'price' is not in the dataset.")
    
X = data.drop('price', axis=1)  # Features
y = data['price']                # Target variable

# One-Hot Encoding for categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (only for numeric columns)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train Multiple Regression Models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)

# Random Forest Regressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

#  Make Predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_dt = dt_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

#  Evaluate Models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Evaluate each model
mse_lin, r2_lin = evaluate_model(y_test, y_pred_lin)
mse_dt, r2_dt = evaluate_model(y_test, y_pred_dt)
mse_rf, r2_rf = evaluate_model(y_test, y_pred_rf)

print(f'Linear Regression: MSE={mse_lin}, R²={r2_lin}')
print(f'Decision Tree: MSE={mse_dt}, R²={r2_dt}')
print(f'Random Forest: MSE={mse_rf}, R²={r2_rf}')

#  Visualize Results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lin)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dt)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Decision Tree Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Random Forest Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Load Dataset
# Replace 'housing_data.csv' with your dataset path
data = pd.read_csv('housing_data.csv')

# Data Preprocessing
print(data.head())
print(data.info())

# Handle Missing Values
if data.isnull().sum().any():
    data.fillna(data.mean(numeric_only=True), inplace=True)  # Use mean only for numeric columns

# Feature Selection
if 'price' not in data.columns:
    raise KeyError("The target variable 'price' is not in the dataset.")
    
X = data.drop('price', axis=1)  # Features
y = data['price']                # Target variable

# One-Hot Encoding for categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (only for numeric columns)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train Multiple Regression Models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)

# Random Forest Regressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

#  Make Predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_dt = dt_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

#  Evaluate Models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Evaluate each model
mse_lin, r2_lin = evaluate_model(y_test, y_pred_lin)
mse_dt, r2_dt = evaluate_model(y_test, y_pred_dt)
mse_rf, r2_rf = evaluate_model(y_test, y_pred_rf)

print(f'Linear Regression: MSE={mse_lin}, R²={r2_lin}')
print(f'Decision Tree: MSE={mse_dt}, R²={r2_dt}')
print(f'Random Forest: MSE={mse_rf}, R²={r2_rf}')

#  Visualize Results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lin)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dt)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Decision Tree Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.title('Random Forest Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.tight_layout()
plt.show()
