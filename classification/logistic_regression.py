# Logistic Regression# Data Preprocessing# Import Librariesimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import StandardScaler# Import Datadf = pd.read_csv('Social_Network_Ads.csv')# Extract Variablesx = df.iloc[:, [2, 3]].valuesy = df.iloc[:, 4].values# Train and Test Datax_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .25, random_state=0)# Scale Datasc_x = StandardScaler()x_train = sc_x.fit_transform(x_train)x_test = sc_x.transform(x_test)# Logistic Regression on Training Setfrom sklearn.linear_model import LogisticRegression# Find correlation in training setsclassifier = LogisticRegression(random_state=0)classifier.fit(x_train, y_train)# Predicts results based on test valuesy_predicted = classifier.predict(x_test)# Confusion matrixfrom sklearn.metrics import confusion_matrixcm = confusion_matrix(y_test, y_predicted)