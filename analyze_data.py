import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('HAM10000_metadata.csv')

# Print basic information
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualize the distribution of lesion types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='dx')
plt.title('Distribution of Lesion Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('lesion_distribution.png')
plt.close()

# Print lesion type counts
print("\nLesion type counts:")
print(df['dx'].value_counts())