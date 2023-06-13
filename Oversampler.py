import pandas as pd

df = pd.read_csv('Dataset/filtered.csv')

# Print all column names
print(df.columns.tolist())

if 'BRCA_subtype' in df.columns:
    # Get counts of each subtype
    subtype_counts = df['BRCA_subtype'].value_counts()

    # Print the counts
    print(subtype_counts)
else:
    print("'BRCA_subtype' column is not found in the dataset.")

from imblearn.over_sampling import RandomOverSampler

# Initialize oversampler
ros = RandomOverSampler(random_state=0)

# Apply the over-sampling
X_resampled, y_resampled = ros.fit_resample(df.drop(columns=['BRCA_subtype']), df['BRCA_subtype'])

# Convert resampled data back into dataframe
df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=df.columns.drop('BRCA_subtype')),
                         pd.DataFrame(y_resampled, columns=['BRCA_subtype'])], axis=1)

# Save balanced data
df_balanced.to_csv('Dataset/balanced.csv', index=False)

# Print all column names
print(df_balanced.columns.tolist())

# Get counts of each subtype
subtype_counts = df_balanced['BRCA_subtype'].value_counts()

# Print the counts
print(subtype_counts)




