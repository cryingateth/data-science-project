import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# Read the filtered CSV file
data = pd.read_csv('../Dataset/filtered.csv')

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

#balance the training data set
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

#check that is now balanced
print(sorted(Counter(y_train_ros).items()))

# Define the hyperparameters for DecisionTree
param_dist = {"max_depth": [1,2,3,4,5,6,7,8,9,10,None], "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10], "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X_train_ros, y_train_ros)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# Predict the labels
y_pred = tree_cv.predict(X_test)

print(classification_report(y_test, y_pred))


# Print the accuracy
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(10,7))
sns.heatmap(conf_mat, annot=True, cmap='Oranges', fmt='d', yticklabels=['Basal', 'Her2', 'LumA', 'LumB', 'Normal'], xticklabels=['Basal', 'Her2', 'LumA', 'LumB', 'Normal'])
plt.title('Confusion Matrix of the Decision Tree classifier')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Save the plot to a file
plt.savefig('../Confusion_matrix/confusion_matrix_dec_trees.jpg')
plt.show()

# Predict the NanSet
nan_data = pd.read_csv('../Dataset/NanSet.csv')
nan_data = nan_data.drop(columns=['BRCA_subtype'])

nan_features = nan_data.select_dtypes(include=[np.number])
nan_scaled_features = scaler.transform(nan_features)
nan_predictions = tree_cv.predict(nan_scaled_features)

# Print the predictions
print("Predictions for NanSet: ", nan_predictions)

# Print the count of predictions for each class
unique_classes, counts = np.unique(nan_predictions, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"Predicted instances for class {cls}: {count}")
