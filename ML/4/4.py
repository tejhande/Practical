import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

# Load the dataset
df = pd.read_csv("/content/diabetes.csv")

# Replace 0s with NaN for specific columns and fill with the mean
zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)  # Replace 0s with NaN
    mean = int(df[column].mean(skipna=True))   # Calculate mean excluding NaNs
    df[column] = df[column].replace(np.NaN, mean)  # Replace NaN with mean value

# Feature matrix and target vector
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Standardize the features
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Initialize and fit the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# Predict on the test data
y_pred = knn.predict(x_test)

# Display predictions
print("Predictions:", y_pred)

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Reds')
ax.set_title("Seaborn Confusion Matrix")
ax.set_xlabel("Predicted values")
ax.set_ylabel("Actual values")
plt.show()

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Precision and Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Error Rate
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate}")
