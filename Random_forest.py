import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, precision_score, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns

# Read the balanced CSV file
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

#balance the train set
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

#check that is now balanced
print(sorted(Counter(y_train_ros).items()))

# Create the parameter grid 
#param_dist = {
#    'bootstrap': [True, False],
#    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 #   'max_features': ['auto', 'sqrt'],
 #   'min_samples_leaf': [1, 2, 4],
 #   'min_samples_split': [2, 5, 10],
  #  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
#}

param_dist = {
    'bootstrap': [False],
    'max_depth': [50],
    'max_features': ['sqrt'],
    'min_samples_leaf': [2],
    'min_samples_split': [2],
    'n_estimators': [200]
}
# Create a base model
rf = RandomForestClassifier()

# Instantiate the randomized search model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the randomized search model to the data
random_search.fit(X_train_ros, y_train_ros)

# Get the best parameters
best_params = random_search.best_params_
print(f'\nBest parameters: {best_params}')

# Train the random forest classifier with best parameters
random_forest = RandomForestClassifier(**best_params)
random_forest.fit(X_train_ros, y_train_ros)

# Predict the test set labels using random forest
y_pred_random_forest = random_forest.predict(X_test)

# Calculate the accuracy of the random forest classifier
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
precision = precision_score(y_test, y_pred_random_forest, average = 'macro')
f1 = f1_score(y_test, y_pred_random_forest, average = 'macro')
recall = recall_score(y_test, y_pred_random_forest, average = 'macro')

print("accurc", accuracy_random_forest, 'precision', precision, 'f1 =', f1, 'reca', recall)

print('-----------------------')

print(f'\nAccuracy of random forest classifier: {accuracy_random_forest:.2f}')

# Print Confusion matrix and classification report for Random Forest Classifier
cm = confusion_matrix(y_test, y_pred_random_forest)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', yticklabels=['Basal', 'Her2', 'LumA', 'LumB', 'Normal'], xticklabels=['Basal', 'Her2', 'LumA', 'LumB', 'Normal'])
plt.title('Confusion matrix of the Random Forest classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix/confusion_matrix_random_forest.jpg')
plt.show()
print(classification_report(y_test, y_pred_random_forest))

# Read the new CSV file
nan_data = pd.read_csv('Dataset/NanSet.csv')

# Define the features for NanSet.csv, assuming they have the same column names
nan_features = nan_data.drop(columns=['BRCA_subtype']) if 'BRCA_subtype' in nan_data.columns else nan_data

# Remove non-numerical features
numerical_nan_features = nan_features.select_dtypes(include=[np.number])


# Standardize the feature matrix for NanSet.csv
nan_scaled_features = scaler.transform(numerical_nan_features)  # Notice use of transform instead of fit_transform

# Predict the labels for NanSet.csv using the trained Random Forest Classifier
nan_pred_random_forest = random_forest.predict(nan_scaled_features)

print(nan_pred_random_forest[0:10])

# Print the count of predicted instances for each class
unique_classes, counts = np.unique(nan_pred_random_forest, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Predicted instances for class {cls}: {count}")


print("-----------------")
coh_kap = cohen_kappa_score(y_test, y_pred_random_forest)
print(coh_kap)
