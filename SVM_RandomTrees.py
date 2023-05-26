import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, f1_score, confusion_matrix, recall_score)

# Read the filtered CSV file
data = pd.read_csv('../TCGA/filtered.csv')
outcome = pd.read_csv('../TCGA/combined_dataset.csv')
# Define the features and target columns
features = data.drop(columns=['BRCA_subtype'])

original_table = pd.read_csv('../TCGA/outcome.csv')
non_nan_orginal = original_table.dropna()
# Remove non-numerical features
numerical_features = features.select_dtypes(include=[np.number])
target = data['BRCA_subtype']

# Standardize the feature matrix
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Train the decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Train the SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Predict the test set labels using decision tree
y_pred_decision_tree = decision_tree.predict(X_test)

# Predict the test set labels using SVM
y_pred_svm = svm.predict(X_test)

#predicting the cancer patients in outcome table
prediction_sub = outcome[outcome['BRCA_subtype'].isna()] #code for the missing values
wanted_gene_subtypes = prediction_sub.select_dtypes(include=[np.number])  

y_pred_goal_sub_tree = decision_tree.predict(wanted_gene_subtypes)

y_pred_goal_sub_svm = svm.predict(wanted_gene_subtypes)

def nan_table_fill(data, y_pred):
    prediction_sub = data[data['BRCA_subtype'].isna()]
    missing = prediction_sub['BRCA_subtype'].isna()
    prediction_sub.loc[missing, 'BRCA_subtype']=y_pred
    return prediction_sub

prediction_sub_tree = nan_table_fill(outcome, y_pred_goal_sub_tree)
prediction_sub_svm = nan_table_fill(outcome, y_pred_goal_sub_svm)

prediction_sub_tree.to_csv('../TCGA/prediction_sub_tree.csv')
prediction_sub_svm.to_csv('../TCGA/prediction_sub_svm.csv')


def fill_total_table(data, y_pred):
    filled_table = data.copy()
    missing_outcome = filled_table['BRCA_subtype'].isna()
    filled_table.loc[missing_outcome, 'BRCA_subtype'] = y_pred
    return filled_table

filled_table_tree = fill_total_table(outcome, y_pred_goal_sub_tree)
filled_table_svm = fill_total_table(outcome, y_pred_goal_sub_svm)

patients_cancer_type_tree = prediction_sub_tree[['Unnamed: 0', 'BRCA_subtype']]
patients_cancer_type_svm = prediction_sub_svm[['Unnamed: 0', 'BRCA_subtype']]

patients_cancer_type_total_svm= pd.concat([patients_cancer_type_svm, non_nan_orginal])
patients_cancer_type_total_tree = pd.concat([patients_cancer_type_tree, non_nan_orginal])


patients_cancer_type_total_svm.to_csv('../TCGA/patients_cancer_type_svm.csv')
patients_cancer_type_total_tree.to_csv('../TCGA/patients_cancer_type_tree.csv')

# Calculate the accuracy of the decision tree classifier
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_decision_tree)
precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average = 'macro')
recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average = 'macro')
f1_decision_tree = f1_score(y_test, y_pred_decision_tree, average='macro')

# Calculate the accuracy of the SVM classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average = 'macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
f1_svm = f1_score(y_test, y_pred_svm, average='macro')

print("DECISION TREE: Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy_decision_tree, precision_decision_tree, recall_decision_tree, f1_decision_tree))
print("-------------------------------------------------------")
print("SVM: Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy_svm, precision_svm, recall_svm, f1_svm))

#print(f'Accuracy of decision tree classifier: {accuracy_decision_tree:.2f}')
#print(f'Accuracy of SVM classifier: {accuracy_svm:.2f}')