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
plt.savefig('confusion_matrix.jpg')

plt.show()
print(classification_report(y_test, y_pred_log_reg))

# Predict the NanSet
nan_data = pd.read_csv('Dataset/NanSet.csv')
nan_features = nan_data.select_dtypes(include=[np.number])
nan_scaled_features = scaler.transform(nan_features)
nan_predictions = log_reg.predict(nan_scaled_features)

# Print the predictions
print("Predictions for NanSet: ", nan_predictions)
