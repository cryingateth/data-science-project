# data-science-project
Group project: TCGA

## Where to save the dataset and outcome table to run the the python codes
"dataset.csv" and "outcome.csv" need to be saved into a local folder titled "Dataset". The folder "Dataset" needs to be saved into the same folder (data-science-project-main), where all the python codes are located. All new generated datasets will also be saved into the folder "Dataset".

## Additional folder
An additional empty folder should be saved in the folder where the python code is located (data-science-project-main). This folder should be called "Confusion_matrix". This is where the confusion matrices will be saved to after running the code. 

## Order of code execution 
1. Dataset_description.py
2. Combined_Dataset.py
3. Filtered_Dataset.py
4. Logistic_regression.py
5. Random_forest.py
6. SVM_classifier.py
7. Decision_trees_classifier.py

For MLP execution (MLP folder):

8. MLP_train.py

## Required Python packages
1. pandas (1.5.3)
2. matplotlib (3.7.1)
3. seaborn (0.12.2)
4. numpy (1.23.5)
5. plotly (5.13.1)
6. skikit-learn (1.2.2)
7. torch (2.0.1)
8. imblearn (0.10.1)

## Data description
Dataset_description.py imports the two datasets (dataset.csv and outcome.csv). The shape, the NaN amount, duplicates amount, dtype are calculated for both datasets. The number of patients per BRCA subtypes are also calculated in the outcome set.

## Data preprocessing
Combined_Dataset.py fuses the TCGA-BRCA gene expression (dataset.csv) table together with the outcome (outcome.csv) table and saves the new dataframe as combined_dataset.csv\
Filtered_Dataset.py removes all the NaN values from combined_dataset.csv and saves the new dataframe as filtered.csv. The removed NaN values are also saved in a new file (NanSet.csv).

## Model
Logistic_regression.py performs the logistic regression using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the logistic regression model.\
Decision_trees.py performs the decision tree using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the decision tree model.\
Random_forest.py performs the random forest using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the random forest model.\
SVM.py performs the SVM using the filtered.csv dataset. Oversampling is done after the split. Evaluation metrics are also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the SVM model.

In the MLP folder, architecture.py contains the structure of neural network (layers), and Dataset.py contains the preprocessing of the data before being added in the MLP model. 
MLP_train.py performs MLP on the filtered dataset using the architecture.py and the Dataset.py. Oversampling is done after the split. Evaluation metrics are  also calculated. Labels' prediction for the NaNs is done using the NanSet.csv and the MLP model. 

