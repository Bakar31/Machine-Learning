import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset import section
dataset = pd.read_csv('CarPrice_Assignment.csv')

plt.figure(figsize=(25,20))
sns.countplot(dataset['carbody'])

#Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset['CarName']=encoder.fit_transform(dataset['CarName'])
dataset['fuelsystem']=encoder.fit_transform(dataset['fuelsystem'])
dataset['cylindernumber']=encoder.fit_transform(dataset['cylindernumber'])
dataset['enginetype']=encoder.fit_transform(dataset['enginetype'])
dataset['enginelocation']=encoder.fit_transform(dataset['enginelocation'])
dataset['drivewheel']=encoder.fit_transform(dataset['drivewheel'])
dataset['carbody']=encoder.fit_transform(dataset['carbody'])
dataset['doornumber']=encoder.fit_transform(dataset['doornumber'])
dataset['aspiration']=encoder.fit_transform(dataset['aspiration'])
dataset['fueltype']=encoder.fit_transform(dataset['fueltype'])

plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(), annot = True, cmap = "RdYlGn")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 25].values

#Splitting dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 7)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print("Train Accuracy:", regressor.score(x_train, y_train))
print("Test Accuracy:", regressor.score(x_test, y_test))