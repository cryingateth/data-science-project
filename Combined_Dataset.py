import pandas as pd

# Read the CSV files
dataset = pd.read_csv('Dataset/dataset.csv')
outcome = pd.read_csv('Dataset/outcome.csv')

# Make sure both DataFrames have the same number of rows
assert dataset.shape[0] == outcome.shape[0], "The number of rows in both CSV files must be the same."

# Stack the DataFrames together horizontally (column-wise)
result = pd.concat([dataset, outcome], axis=1)

# Save the combined DataFrame to a new CSV file
result.to_csv('Dataset/combined_dataset.csv', index=False)
