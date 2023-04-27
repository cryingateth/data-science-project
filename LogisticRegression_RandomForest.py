import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the filtered CSV file
data = pd.read_csv('Dataset/filtered.csv')

# Define the features and target columns
features = data.drop(columns=['BRCA_subtype'])

# Remove non-numerical features
numerical_features = features.select_dtypes(include=[np.number])
target = data['BRCA_subtype']

# Standardize the feature matrix
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Train the random forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predict the test set labels using logistic regression
y_pred_log_reg = log_reg.predict(X_test)

# Predict the test set labels using random forest
y_pred_random_forest = random_forest.predict(X_test)

# Calculate the accuracy of the logistic regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Calculate the accuracy of the random forest classifier
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)

print(f'Accuracy of logistic regression: {accuracy_log_reg:.2f}')
print(f'Accuracy of random forest classifier: {accuracy_random_forest:.2f}')

# Print some predictions using logistic regression
print("Predictions using logistic regression:")
print("True labels: ", list(y_test[:15]))
print("Predicted labels: ", list(y_pred_log_reg[:15]))

# Print some predictions using random forest
print("\nPredictions using random forest:")
print("True labels: ", list(y_test[:15]))
print("Predicted labels: ", list(y_pred_random_forest[:15]))
