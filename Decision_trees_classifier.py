import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Read the filtered CSV file
data = pd.read_csv('Dataset/balanced.csv')

# Define the features and target columns
features = data.drop(columns=['BRCA_subtype'])

# Remove non-numerical features
numerical_features = features.select_dtypes(include=[np.number])
target = data['BRCA_subtype']

# Binarize the output
target = label_binarize(target, classes=np.unique(target))

# Standardize the feature matrix
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Define the hyperparameters for DecisionTreeClassifier
param_dist = {
    "estimator__max_depth": [1, 3, 5, 7, 9, None],
    "estimator__min_samples_leaf": range(1, 10),
    "estimator__criterion": ["gini", "entropy"],
}

# Instantiate a Decision Tree classifier
tree = OneVsRestClassifier(DecisionTreeClassifier())

# Instantiate the RandomizedSearchCV object
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# Predict the labels
y_score = tree_cv.predict(X_test)

# Plot the confusion matrix and save it to a file
plot_confusion_matrix(tree_cv, X_test, y_test.argmax(axis=1), cmap=plt.cm.Oranges, values_format='.0f')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.jpg')
plt.show()


# Read the NanSet CSV file
nan_data = pd.read_csv('Dataset/NanSet.csv')

# Define the features
nan_features = nan_data.drop(columns=['BRCA_subtype'])  # assuming the target column is 'BRCA_subtype'

# Remove non-numerical features
nan_numerical_features = nan_features.select_dtypes(include=[np.number])

# Use the same scaler to standardize the nan dataset features
nan_scaled_features = scaler.transform(nan_numerical_features)

# Predict the labels for NanSet data
nan_predictions = tree_cv.predict(nan_scaled_features)

# Print the predictions
print("Predictions for NanSet data: ", nan_predictions)

