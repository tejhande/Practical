# 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/uber_rides.csv")
df.head()

# 2
df.info()

# 3
df.shape

# 4
df.isnull().sum()

# 5
df.dropna(inplace=True)
df.isnull().sum()

# 6
df.drop(labels='Unnamed: 0', axis=1,inplace=True)
df.drop(labels='key', axis=1, inplace=True)
df.head()

# 7
df["pickup_datetime"]=pd.to_datetime(df["pickup_datetime"])
df.dtypes

# 8
df.describe()

# 9
import seaborn as sns
sns.distplot(df['fare_amount'])

# 10
sns.distplot(df['pickup_latitude'])

# 11
sns.distplot(df['pickup_longitude'])

# 12
sns.distplot(df['dropoff_longitude'])

# 13
sns.distplot(df['dropoff_latitude'])

# 14
def find_outliers_IQR(df):
  q1=df.quantile(0.25)
  q3=df.quantile(0.75)
  IQR=q3-q1
  outliers=df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
  return outliers
outliers=find_outliers_IQR(df["fare_amount"])
print("number of outliers: "+str(len(outliers)))
print("max outlier value: "+str(outliers.max()))
print("number of outliers: "+str(outliers.min()))
outliers

# 15
outliers = find_outliers_IQR(df[["passenger_count","fare_amount"]])
outliers

# 16
upper_limit = df['fare_amount'].mean() + 3*df['fare_amount'].std()
print(upper_limit)
lower_limit = df['fare_amount'].mean() + 3*df['fare_amount'].std()
print(lower_limit)

# 17
import seaborn as sns
import matplotlib.pyplot as plt
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# 18
import pandas as pd
import calendar
# Example: Creating a DataFrame (you'll likely already have df)
# df = pd.read_csv('your_data.csv')  # Replace with your actual data loading method
# Check if 'pickup_datetime' exists
if 'pickup_datetime' in df.columns:
    # Convert 'pickup_datetime' to datetime if not already
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    # Extract components
    df['day'] = df['pickup_datetime'].dt.day
    df['hour'] = df['pickup_datetime'].dt.hour
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['weekday'] = df['pickup_datetime'].dt.day_name()
    # Drop 'pickup_datetime' column
    df.drop('pickup_datetime', axis=1, inplace=True)
    # Map weekdays to numeric values
    df['weekday'] = df['weekday'].map({
        'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
        'Thursday': 4, 'Friday': 5, 'Saturday': 6
    })
else:
    print("Column 'pickup_datetime' not found in DataFrame.")
# View the first few rows of the DataFrame
print(df.head())

# 19
df.info()

# 20
from sklearn.model_selection import train_test_split
x=df.drop("fare_amount", axis=1)
x

# 21
from sklearn.model_selection import train_test_split
# Splitting the data into features (X) and target (y)
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# View the first few rows of the training features (x_train)
print(x_train.head())

# 22
x_test.head()

# 23
y_train.head()

# 24
y_test.head()
