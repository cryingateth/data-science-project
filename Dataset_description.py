import pandas as pd

dataset = pd.read_csv('Dataset/dataset.csv')
outcome = pd.read_csv('Dataset/outcome.csv')

print(dataset.head())
print("_"*25)
print(outcome.head())

print("Number of columns of dataset: ", len(dataset.columns))
print("Number of columns ot outcome table: ", len(outcome.columns))

print("Number of rows of dataset: ", len(dataset.index))
print("Number of rows of outcome: ", len(outcome.index))

number_nan_outcome = outcome.dropna()

nan = len(outcome.index)-len(number_nan_outcome.index)

print("There are", nan, "empty rows in the outcome table that need to be filled.")

duplicates = sum(dataset.duplicated(subset=None, keep='first'))
print("Number of duplicates in dataset: ", duplicates)
print("Number of patient duplicates in outcome table: ", sum(outcome.duplicated(subset=None, keep='first')))

subtypes = number_nan_outcome['BRCA_subtype'].unique()
print("The BRCA subtypes: ", subtypes)

print(outcome['BRCA_subtype'].value_counts())

print("Dataset data types")
print(dataset.dtypes)
print("_"*25)
print("Outcome table data types")
print(outcome.dtypes)