#CS7641: Machine Learning
#Fall 2018
#aelkugia3

#Boosting

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

dataset = pd.read_csv("/Users/aelkugia/Documents/GeorgiaTech/CS7641/aelkugia3/aelkugia3/bill_authentication.csv")  #Update with dataset

dataset.shape #identify number of rows/columns in dataset
dataset.head() #inspect the first five records of the dataset

# Dividing data into attributes and labels
# X -> attribute set
# y -> corresponding labels

X = dataset.drop('Class', axis=1) # X variable contains all columns from the dataset, except "Class"
y = dataset['Class'] # Y variable contains the values from the "Class Column"

# Random splitting data into training and test sets  

stats = []
x = []

for i in np.arange(0.2, 1.0, 0.2):  

	from sklearn.model_selection import train_test_split  
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= i)  #test_size -> 20% data for test set, 80% for training

# Training the decision tree on the training data

	from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

# Following pruning methods, identified the following:
# colsample_bytree = 0.5
# max_depth = 6
# colsample_bylevel = 0.1

	classifier = XGBClassifier(colsample_bytree = 0.5, max_depth = 6, colsample_bylevel = 0.1)  

	classifier.fit(X_train, y_train)  

	start = time.clock()

	y_pred = classifier.predict(X_test)  

	print time.clock()

	print(confusion_matrix(y_test, y_pred))  
	print(classification_report(y_test, y_pred)) 

	stats.append(recall_score(y_test, y_pred))

	x.append((1-i)*100)

plt.plot(x, stats)
plt.title("Training Size vs. Recall Score")
plt.xlabel("Training Size (%)")
plt.ylabel("Recall Score")
plt.show()

# # Set the parameters by cross-validation
# tuned_parameters = [{'max_depth': [2,6], 'colsample_bytree': [0.1,0.5], 'colsample_bylevel' : [0.1,0.5]}]

# clf = GridSearchCV(classifier, tuned_parameters, cv=5)
# clf.fit(X_train, y_train)

# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()


#Two pruning Methods
# max depth and col sample by tree, and level
# col sample = ratio of columns to be used, smaller number of columns more conservative

    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']

    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()

    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print()


#TODO:
#1) Check works first - it works
#2) Add GridSearch for various parameter OR try Random Search



