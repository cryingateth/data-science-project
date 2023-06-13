import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score

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

#balance the train set
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

#check that is now balanced
print(sorted(Counter(y_train_ros).items()))

# Define the hyperparameters for SVC
#param_dist = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001], "kernel": ['linear', 'poly', 'rbf']}
param_dist = {"C": [10], "gamma": [0.001], "kernel": ['linear']}

# Instantiate a SVC classifier
svc = SVC()

# Instantiate the RandomizedSearchCV object
svc_cv = RandomizedSearchCV(svc, param_dist, cv=5)

# Fit it to the data
svc_cv.fit(X_train_ros, y_train_ros)

# Print the tuned parameters and score
print("Tuned SVC Parameters: {}".format(svc_cv.best_params_))
print("Best score is {}".format(svc_cv.best_score_))

# Predict the labels
y_pred = svc_cv.predict(X_test)

# Print the accuracy
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

# Print the classification report
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, cmap='Oranges', fmt='d', yticklabels=['Basal', 'Her2', 'LumA', 'LumB', 'Normal'], xticklabels=['Basal', 'Her2', 'LumA', 'LumB', 'Normal'])
plt.title('Confusion Matrix of the SVM classifier')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Save the plot to a file
plt.savefig('Confusion_matrix/confusion_matrix_svm.jpg')
plt.show()

# Read the new CSV file
nan_data = pd.read_csv('Dataset/NanSet.csv')

# Define the features for NanSet.csv, assuming it has the same column names
nan_features = nan_data.drop(columns=['BRCA_subtype']) if 'BRCA_subtype' in nan_data.columns else nan_data

# Remove non-numerical features
numerical_nan_features = nan_features.select_dtypes(include=[np.number])

# Standardize the feature matrix for NanSet.csv
nan_scaled_features = scaler.transform(numerical_nan_features)  # Notice use of transform instead of fit_transform

# Predict the labels for NanSet.csv
nan_pred_svc = svc_cv.predict(nan_scaled_features)

# Print the count of predicted instances for each class
unique_classes, counts = np.unique(nan_pred_svc, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Predicted instances for class {cls}: {count}")

print(nan_pred_svc[0:10])


print("-----------------")
coh_kap = cohen_kappa_score(y_test, y_pred)
print(coh_kap)
