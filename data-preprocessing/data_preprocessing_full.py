# Data Preprocessing

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import Data
df = pd.read_csv('Data.csv')

# Extract Variables
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Impute missing values with mean
imputer = SimpleImputer()
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode categorical data into binary representation
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.str)

# Encode decision from text to number
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .2, random_state=0)

# Scale Data
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)