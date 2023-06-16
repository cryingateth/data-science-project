# data-science-project
Group project: TCGA

## Data description


## Data preprocessing
Combined_Dataset.py fuses the TCGA-BRCA gene expression table together with the outcome table and saves the new dataframe as combined_dataset.csv\
Filtered_Dataset.py removes all the NaN values from combined_dataset.csv and saves the new dataframe as filtered.csv

## Model:
Logistic_regression.py performs the logistic regression using the filtered.csv dataset.\
Decision_trees.py performs the decision tree using the filtered.csv dataset.\
Random_forest.py performs the random forest using the filtered.csv dataset.\
SVM.py performs the SVM using the filtered.csv dataset.\

In the MLP folder, architecture.py does something, and Dataset.py does something else that is needed in MLP_train.py.
MLP_train.py performs MLP on the filtered dataset

