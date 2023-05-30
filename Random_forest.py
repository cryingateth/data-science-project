import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Create the parameter grid 
param_dist = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

# Create a base model
rf = RandomForestClassifier()

# Instantiate the randomized search model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the randomized search model to the data
random_search.fit(X_train, y_train)

# Print the complete results
for mean_test_score, params in zip(random_search.cv_results_["mean_test_score"], random_search.cv_results_["params"]):
    print(f"Mean test score: {mean_test_score:.3f} for parameters {params}")

# Get the best parameters
best_params = random_search.best_params_
print(f'\nBest parameters: {best_params}')

# Train the random forest classifier with best parameters
random_forest = RandomForestClassifier(**best_params)
random_forest.fit(X_train, y_train)

# Predict the test set labels using random forest
y_pred_random_forest = random_forest.predict(X_test)

# Calculate the accuracy of the random forest classifier
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)

print(f'\nAccuracy of random forest classifier: {accuracy_random_forest:.2f}')

# Print Confusion matrix and classification report for Random Forest Classifier
cm = confusion_matrix(y_test, y_pred_random_forest)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix of the Random Forest classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print(classification_report(y_test, y_pred_random_forest))
