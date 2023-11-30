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

