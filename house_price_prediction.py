import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import tarfile

# Define the download URL and the destination path
url = "https://ndownloader.figshare.com/files/5976036"
dataset_dir = '/Users/tarunksaini/Codex/House_ptice/house_price_env/CaliforniaHousing'

# Download the dataset file
response = requests.get(url)
dataset_file_path = os.path.join(dataset_dir, "california_housing.tar.gz")

# Check if the directory exists, if not, create it
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Save the file locally
with open(dataset_file_path, "wb") as f:
    f.write(response.content)

# Extract the file if not already extracted
if not os.path.exists(os.path.join(dataset_dir, 'california_housing')):
    with tarfile.open(dataset_file_path, "r:gz") as tar:
        tar.extractall(path=dataset_dir)

# Now load the dataset using fetch_california_housing
data = fetch_california_housing(data_home=dataset_dir)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

# Checking for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Descriptive statistics for understanding distributions
print("\nDescriptive statistics of the dataset:")
print(df.describe())

# Feature and target variable
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nMean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plotting Actual vs Predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
