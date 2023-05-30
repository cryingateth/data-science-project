import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV

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

# Create a logistic regression model
log_reg = LogisticRegression()

# Define the parameter distribution
param_dist = {'max_iter': range(100, 10000, 100)}

# Perform randomized search
random_search = RandomizedSearchCV(log_reg, param_distributions=param_dist, 
                                   n_iter=20, scoring='neg_log_loss', cv=5, 
                                   random_state=42, n_jobs=-1, verbose=3)

random_search.fit(X_train, y_train)

# Print the best number of iterations
print(f'Best max_iter: {random_search.best_params_["max_iter"]}')

# Predict the test set labels using logistic regression
y_pred_log_reg = random_search.predict(X_test)

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
