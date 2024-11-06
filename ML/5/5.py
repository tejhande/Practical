# Install Yellowbrick if not already installed (Run this in a separate cell in Colab/Jupyter)
# !pip install yellowbrick

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter
import warnings

# Ignore future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("/content/Auto Sales data.csv")  # Adjust the path if necessary

# Initial Data Exploration
print("First few rows of the data:")
print(df.head())

print("\nShape of the dataset:")
print(df.shape)

print("\nChecking for missing values:")
print(df.isnull().sum())

# Data visualization (Countplot for 'STATUS' and 'PRODUCTLINE')
sns.countplot(data = df, x = 'STATUS')
plt.show()

sns.histplot(x= 'SALES', hue='PRODUCTLINE', data=df, element="poly")
plt.show()

# Check unique values in 'PRODUCTLINE'
print("\nUnique values in 'PRODUCTLINE':")
print(df['PRODUCTLINE'].unique())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Check dataset info after dropping duplicates
print("\nData info after dropping duplicates:")
df.info()

# Visualizing categorical features
list_cat = df.select_dtypes(include=['object']).columns.tolist()

# Plot countplots for all categorical columns
for i in list_cat:
    sns.countplot(data = df, x=i)
    plt.xticks(rotation=90)
    plt.show()

# Label encoding for categorical columns
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in list_cat:
    df[i] = le.fit_transform(df[i])

# Info after label encoding
df.info()

# Convert 'SALES' to integer type
df['SALES'] = df['SALES'].astype(int)

# Check dataset info again
df.info()

# Summary statistics of numerical columns
print("\nSummary statistics:")
print(df.describe())

# Select features for clustering
x = df[['SALES', 'PRODUCTCODE']]  # Using 'SALES' and 'PRODUCTCODE' for clustering

# Scaling the features (important for KMeans)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Elbow method using Yellowbrick for optimal number of clusters
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 12))

# Fit the visualizer and show the elbow plot
visualizer.fit(x_scaled)
visualizer.show()

# Based on the elbow plot, choose the optimal number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
kmeans.fit(x_scaled)

# Displaying results
print("\nCluster Labels:")
print(kmeans.labels_)

print("\nInertia (within-cluster sum of squares):")
print(kmeans.inertia_)

print("\nNumber of iterations to converge:")
print(kmeans.n_iter_)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Counting the number of data points in each cluster
print("\nCluster Distribution (size of each cluster):")
print(Counter(kmeans.labels_))

# Optionally, add cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Display the dataframe with cluster labels
print("\nData with cluster labels:")
print(df.head())

# Optional: Visualize clusters (only if you have 2 features, or use PCA for dimensionality reduction)
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Sales')
plt.ylabel('Product Code')
plt.title('KMeans Clustering')
plt.show()
