#CS7641: Machine Learning
#Fall 2018
#aelkugia3

#Decision Tree with Pruning
#Citation: https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
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

	start = time.clock() 

	from sklearn.model_selection import train_test_split  

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= i)  #test_size -> 20% data for test set, 80% for training

# Training the decision tree on the training data

	from sklearn.tree import DecisionTreeClassifier 
	from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

	classifier = DecisionTreeClassifier(max_depth = 6)  

	classifier.fit(X_train, y_train)  

	y_pred = classifier.predict(X_test)  

	print(confusion_matrix(y_test, y_pred))  
	print(classification_report(y_test, y_pred))  

	stats.append(recall_score(y_test, y_pred))

	x.append((1-i)*100)

plt.plot(x, stats)
plt.title("Training Size vs. Recall Score")
plt.xlabel("Training Size (%)")
plt.ylabel("Recall Score")
plt.show()


# Identifyin Max Depth

# stats = []

# for i in range(1, 10):
	 
# 	classifier = DecisionTreeClassifier(max_depth = i)  
# 	classifier.fit(X_train, y_train)  

# 	print(i)

# 	y_pred = classifier.predict(X_test)  
# 	print(confusion_matrix(y_test, y_pred))  
# 	print(classification_report(y_test, y_pred))  

# 	stats.append(recall_score(y_test, y_pred))

# plt.plot(range(1,10), stats)
# plt.title("Decision Tree w/ Pruning")
# plt.xlabel("Max Depth of Tree")
# plt.ylabel("Recall Score")
# plt.show()

# Evaluating the algorithm (metrics including confusion matrix, precision, recall, and F1 score)

