import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


from sklearn.model_selection import cross_val_score

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

# Initial setting
best_score = -np.inf
best_max_iter = None

# For different number of iterations
for max_iter in range(100, 10000, 100):
    # Create a logistic regression model
    log_reg = LogisticRegression(max_iter=max_iter)
    
    # Perform cross-validation
    scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='neg_log_loss')

    # Calculate the mean score
    mean_score = np.mean(scores)

    # If the score is better than the current best score, update the best score and best max_iter
    if mean_score > best_score:
        best_score = mean_score
        best_max_iter = max_iter

# Print the best number of iterations
print(f'Best max_iter: {best_max_iter}')

# Train the logistic regression model with the best number of iterations
log_reg = LogisticRegression(max_iter=best_max_iter)
log_reg.fit(X_train, y_train)

# Predict the test set labels using logistic regression
y_pred_log_reg = log_reg.predict(X_test)

# Calculate the accuracy of the logistic regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(f'Accuracy of logistic regression: {accuracy_log_reg:.2f}')

# Print Confusion matrix and classification report for Logistic Regression
cm = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix of the Logistic Regression classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(y_test, y_pred_log_reg))
