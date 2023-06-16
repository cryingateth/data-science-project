# data-science-project
Group project: TCGA

## Data description
Dataset_description.py imports the two datasets (dataset.csv and outcome.csv). The shape, the NaN amount, duplicates amount, dtype are calculated for both datasets. The number of patients per BRCA subtypes was also calculated in the outcome set.

## Data preprocessing
Combined_Dataset.py fuses the TCGA-BRCA gene expression (dataset.csv) table together with the outcome (outcome.csv) table and saves the new dataframe as combined_dataset.csv\
Filtered_Dataset.py removes all the NaN values from combined_dataset.csv and saves the new dataframe as filtered.csv. The removed NaN values are also saved in a new file (NanSet.csv).

## Model:
Logistic_regression.py performs the logistic regression using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the logistic regression model.\
Decision_trees.py performs the decision tree using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the decision tree model.\
Random_forest.py performs the random forest using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the random forest model.\
SVM.py performs the SVM using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the SVM model.\

In the MLP folder, architecture.py contains the structure of neural network (layers), and Dataset.py contains the processing of the data before being added in the MLP model. 
MLP_train.py performs MLP on the filtered dataset using the architecture.py and the Dataset.py. Oversampling is done after the split. Evaluation metrics are  also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the MLP model. 

