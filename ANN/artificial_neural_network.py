# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data into binary representation
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
country_encoder = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(country_encoder.fit_transform(x), dtype = np.str)
gender_encoder = ColumnTransformer([('encoder', OneHotEncoder(), [4])], remainder='passthrough')
x = np.array(gender_encoder.fit_transform(x), dtype = np.str)

# Deletes the third country column to avoid dummy variable trap
x = np.delete(x, 2, 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > .5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # Initialize ANN
    classifier = Sequential()
    
    # Input and first hidden Layer
    classifier.add(Dense(activation = "relu", input_dim = 12, units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    
    # Second hidden layer
    classifier.add(Dense(activation = "relu", input_dim = 12, units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dropout(rate = 0.1))
    
    # Output layer
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X= x_train, y = y_train, cv = 10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
# Tuning

# Evaluation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    # Initialize ANN
    classifier = Sequential()
    
    # Input and first hidden Layer
    classifier.add(Dense(activation = "relu", input_dim = 12, units = 6, kernel_initializer = 'uniform'))
    
    # Second hidden layer
    classifier.add(Dense(activation = "relu", input_dim = 12, units = 6, kernel_initializer = 'uniform'))
    
    # Output layer
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv= 10)
grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
