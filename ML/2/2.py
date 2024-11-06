import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
df=pd.read_csv("/content/emails.csv")
df.head()

df.info()

df.isnull().sum()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Assuming `df` is your DataFrame
X = df.iloc[:, 1:-1].values  # Features (all columns except the first and last)
y = df.iloc[:, -1].values    # Target (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Initialize the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier with the training data
classifier.fit(X_train, y_train)

# Optionally, evaluate the classifier with the test data
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Predict the labels for the test set
y_pred = classifier.predict(X_test)  # Use X_test instead of x_test

# Import necessary metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute accuracy score (optional)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

from sklearn.metrics import classification_report
cl_report=classification_report(y_test,y_pred)
print(cl_report)

print ("Accuracy Score of KNN :",accuracy_score(y_pred,y_test))
