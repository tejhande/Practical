import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Correct import for matplotlib

# Load the dataset
df = pd.read_csv("/content/emails.csv")

# Display the first few rows and data info
print(df.head())
df.info()

# Check if there are any non-numeric columns or missing values in the dataset
print(df.isnull().sum())

# Assuming df has features in all columns except the first and last one, and the last column is the target
X = df.iloc[:, 1:-1].values  # Features (all columns except the first and last)
y = df.iloc[:, -1].values    # Target (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Initialize the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier with the training data
classifier.fit(X_train, y_train)

# Evaluate the classifier with the test data
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Import necessary metrics for evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute accuracy score (optional)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

# Classification report
cl_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(cl_report)
