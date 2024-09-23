import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (mean = 0, variance = 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the deep learning model
model = Sequential()

# Input layer and first hidden layer with 128 neurons
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# Second hidden layer with 64 neurons
model.add(Dense(64, activation='relu'))

# Third hidden layer with 32 neurons
model.add(Dense(32, activation='relu'))

# Output layer (for regression, no activation function)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse}")

# Plot the loss curve for training and validation
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Data: {mae}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error on Test Data: {rmse}")

# R-squared (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)
print(f"R-squared on Test Data: {r2}")
# Define a function to calculate accuracy as the percentage of predictions within a certain tolerance
def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    # Calculate the absolute differences
    differences = np.abs(y_true - y_pred.reshape(-1))
    
    # Calculate the number of predictions within the tolerance
    accurate_predictions = np.sum(differences <= tolerance)
    
    # Calculate the accuracy as a percentage
    accuracy = accurate_predictions / len(y_true) * 100
    return accuracy

# Calculate accuracy with a tolerance of 0.1
accuracy = calculate_accuracy(y_test, y_pred, tolerance=0.1)
print(f"Accuracy within ±0.1 tolerance: {accuracy:.2f}%")
