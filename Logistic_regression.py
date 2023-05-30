import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Read the filtered CSV file
data = pd.read_csv('Dataset/balanced.csv')

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

# Create a logistic regression model with a large number of maximum iterations
log_reg = LogisticRegression(max_iter=10000)

# Fit the model
log_reg.fit(X_train, y_train)

# Predict the test set labels
y_pred_log_reg = log_reg.predict(X_test)

# Calculate the accuracy of the logistic regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(f'Accuracy of logistic regression: {accuracy_log_reg:.2f}')

# Print Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_log_reg)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
plt.title('Confusion matrix of the Logistic Regression classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Save the figure
plt.savefig('confusion_matrix_lr.jpg')
print(classification_report(y_test, y_pred_log_reg))

# Read the new CSV file
nan_data = pd.read_csv('Dataset/NanSet.csv')

# Define the features and target columns for NanSet.csv, assuming they have the same column names
nan_features = nan_data.drop(columns=['BRCA_subtype'])

# Remove non-numerical features
numerical_nan_features = nan_features.select_dtypes(include=[np.number])
nan_target = nan_data['BRCA_subtype']

# Standardize the feature matrix for NanSet.csv
nan_scaled_features = scaler.transform(numerical_nan_features)  # Notice use of transform instead of fit_transform

# Predict the labels for NanSet.csv
nan_pred_log_reg = log_reg.predict(nan_scaled_features)

# Print the count of predicted instances for each class
unique_classes, counts = np.unique(nan_pred_log_reg, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Predicted instances for class {cls}: {count}")



