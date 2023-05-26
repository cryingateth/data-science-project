import pandas as pd

log_reg_data = pd.read_csv('../TCGA/prediction_sub_LR.csv')
rf_data = pd.read_csv('../TCGA/prediction_sub_RF.csv')
svm_data = pd.read_csv('../TCGA/prediction_sub_svm.csv')
tree_data = pd.read_csv('../TCGA/prediction_sub_tree.csv')

print("LR:", log_reg_data['BRCA_subtype'].value_counts())
print("RF:", rf_data['BRCA_subtype'].value_counts())
print('SVM:', svm_data['BRCA_subtype'].value_counts())
print("Tree:", tree_data['BRCA_subtype'].value_counts())
