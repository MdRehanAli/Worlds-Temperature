#Worlds Temperature Prediction

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #Sklearn for the main model & LinearRegression model for regression

# Load the dataset
data = pd.read_csv('D:\Programming\Python\Worlds-Temperature\Worlds Temperature.csv')

print(data.head())

# Converted Data from Farenheit to Celcius
data['Average_Temperature_in_C'] = round(((data['Average_Temperature_in_F'] - 32) / 1.8), 2)
print(data.head())

# Get data
X = data[['Year']].values
y = data[['Average_Temperature_in_C']].values

print(X, y)

# Finding the Min and Max temperatures from the dataset.
min_temperature_date = data.loc[data['Average_Temperature_in_C'].idxmin(), 'Year']
min_temperature_value = data['Average_Temperature_in_C'].min()
max_temperature_date = data.loc[data['Average_Temperature_in_C'].idxmax(), 'Year']
max_temperature_value = data['Average_Temperature_in_C'].max()

print(f"The minimum temperature of {min_temperature_value}°C occurred on {min_temperature_date}.")
print(f"The maximum temperature of {max_temperature_value}°C occurred on {max_temperature_date}.")

# Split to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train, X_test, y_train, y_test)

# Training model
reg = LinearRegression()

print(reg.fit(X_train, y_train))

# Testing
print(reg.predict([[2123]]))


