import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)

# Read the filtered CSV file

def fill_table(df, y_pred, name, column_name):
    missing = df[column_name].isna()
    name = df.copy()
    name.loc[missing, column_name] = y_pred
    return name

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
# Fanny: splitting multiple times results in better accuracy
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Train the random forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predict the test set labels using logistic regression
y_pred_log_reg = log_reg.predict(X_test)

# Predict the test set labels using random forest
y_pred_random_forest = random_forest.predict(X_test)

#fill out outcome table/NaN values


prediction_sub = outcome[outcome['BRCA_subtype'].isna()] #code for the missing values
wanted_gene_subtypes = prediction_sub.select_dtypes(include=[np.number])  

y_pred_goal_sub_LR = log_reg.predict(wanted_gene_subtypes)

y_pred_goal_sub_RF = random_forest.predict(wanted_gene_subtypes)


def nan_table_fill(data, y_pred):
    prediction_sub = data[data['BRCA_subtype'].isna()]
    missing = prediction_sub['BRCA_subtype'].isna()
    prediction_sub.loc[missing, 'BRCA_subtype']=y_pred
    return prediction_sub


#here is good
#prediction table with just cancers that were missing before
#prediction_sub_LR = prediction_sub.copy()
#prediction_sub_RF = prediction_sub.copy()
#missing = prediction_sub_LR['BRCA_subtype'].isna()
#prediction_sub_LR.loc[missing, 'BRCA_subtype'] = y_pred_goal_sub_LR
#prediction_sub_RF.loc[missing, 'BRCA_subtype'] = y_pred_goal_sub_RF

#the prediction data frames have all genes and the cancer type. It is 262 rows aka same number of original NaN values

prediction_sub_LR = nan_table_fill(outcome, y_pred_goal_sub_LR)
prediction_sub_RF = nan_table_fill(outcome, y_pred_goal_sub_RF)
prediction_sub_LR.to_csv('../TCGA/prediction_sub_LR.csv')
prediction_sub_RF.to_csv('../TCGA/prediction_sub_RF.csv')


#filled table = total combined table with all cancer NaN values filled

def fill_total_table(data, y_pred):
    filled_table = data.copy()
    missing_outcome = filled_table['BRCA_subtype'].isna()
    filled_table.loc[missing_outcome, 'BRCA_subtype'] = y_pred
    return filled_table

filled_table_LR = fill_total_table(outcome, y_pred_goal_sub_LR)
filled_table_RF = fill_total_table(outcome, y_pred_goal_sub_RF)

#filled_table_LR = outcome.copy()
#filled_table_RF = outcome.copy()
#missing_outcome = filled_table_LR['BRCA_subtype'].isna()
#filled_table_LR.loc[missing_outcome, 'BRCA_subtype'] = y_pred_goal_sub_LR
#filled_table_RF.loc[missing_outcome, 'BRCA_subtype'] = y_pred_goal_sub_RF

print("Number of NaN values in final filled dataset:", filled_table_LR['BRCA_subtype'].isnull().sum())

#patient cancer type is table of only the missing values and the patients now with the correct cancer type
patients_cancer_type_LR = prediction_sub_LR[['Unnamed: 0', 'BRCA_subtype']]
patients_cancer_type_RF = prediction_sub_RF[['Unnamed: 0', 'BRCA_subtype']]

#table with all cancer patients and their diagnoses
patients_cancer_type_total_LR= pd.concat([patients_cancer_type_LR, non_nan_orginal])
patients_cancer_type_total_RF = pd.concat([patients_cancer_type_RF, non_nan_orginal])


patients_cancer_type_total_LR.to_csv('../TCGA/patients_cancer_type_LR.csv')
patients_cancer_type_total_RF.to_csv('../TCGA/patients_cancer_type_RF.csv')

# Calculate the accuracy of the logistic regression model
confusion_log_reg = confusion_matrix(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg, average = 'macro')
f1_log_reg = f1_score(y_test, y_pred_log_reg, average = 'macro')
recall_log_reg = recall_score(y_test, y_pred_log_reg, average = 'macro')


# Calculate the accuracy of the random forest classifier
confusion_random_forest = confusion_matrix(y_test, y_pred_random_forest)
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
precision_random_forest = precision_score(y_test, y_pred_random_forest, average = 'macro')
f1_random_forest = f1_score(y_test, y_pred_random_forest, average = 'macro')
recall_random_forest = recall_score(y_test, y_pred_random_forest, average='macro')

print("LOGISTIC REGRESSION EVALUATION DATA: Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy_log_reg, precision_log_reg, recall_log_reg, f1_log_reg))
print("-----------------------------------------------------------------------------")
print("RANDOM FOREST EVALUATION DATA: Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy_random_forest, precision_random_forest, recall_random_forest, f1_random_forest))


# Print some predictions using logistic regression
print("Predictions using logistic regression:")
print("True labels: ", list(y_test[:15]))
print("Predicted labels: ", list(y_pred_log_reg[:15]))

# Print some predictions using random forest
print("\nPredictions using random forest:")
print("True labels: ", list(y_test[:15]))
print("Predicted labels: ", list(y_pred_random_forest[:15]))

