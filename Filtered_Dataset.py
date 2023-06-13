import pandas as pd

# Read the combined CSV file
combined = pd.read_csv('Dataset/combined_dataset.csv')
print('The shape of combined dataset is:', combined.shape)

# Remove samples with missing or NaN values in the "BRCA_subtype" column
filtered = combined.dropna(subset=['BRCA_subtype'])

# Save the filtered DataFrame to a new CSV file
filtered.to_csv('Dataset/filtered.csv', index=False)
print('The shape of combined dataset after removing NaN is:', filtered.shape)

# Filter out the samples with NaN values
nan_set = combined[combined['BRCA_subtype'].isna()]

# Save the NaN DataFrame to a new CSV file
nan_set.to_csv('Dataset/NanSet.csv', index=False)
print('The shape of NaN dataset is:', nan_set.shape)
