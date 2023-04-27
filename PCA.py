import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the filtered CSV file
data = pd.read_csv('Dataset/filtered.csv')

# Define the features and target columns
features = data.drop(columns=['BRCA_subtype'])
target = data['BRCA_subtype']
features = features.iloc[:,1:]

# Identify and print non-numeric feature names
non_numeric_features = features.select_dtypes(include=['object', 'category'])
print("Non-numeric feature names:\n", non_numeric_features.columns)

# Remove non-numeric features
numeric_features = features.select_dtypes(exclude=['object', 'category'])

# Standardize the feature matrix
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

# Perform PCA
pca = PCA()
pca_features = pca.fit_transform(scaled_features)

# Create a scree plot of the explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot of PCA Components')
plt.grid()
plt.show()

# Choose the number of components based on the desired explained variance threshold (e.g., 0.95)
n_components = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1

# Perform PCA with the chosen number of components
pca_optimal = PCA(n_components=n_components)
pca_features_optimal = pca_optimal.fit_transform(scaled_features)

# Create a new DataFrame with the PCA components and the target column
pca_df = pd.DataFrame(pca_features_optimal, columns=[f'PC{i+1}' for i in range(pca_features_optimal.shape[1])])
pca_df['BRCA_subtype'] = target.values

# Save the PCA-transformed DataFrame to a new CSV file
pca_df.to_csv('Dataset/pca_transformed.csv', index=False)
