#CS7641: Machine Learning
#Fall 2018
#aelkugia3

#Support Vector Machines
#Citation: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import time

# import data set

bankdata = pd.read_csv("/Users/aelkugia/Documents/GeorgiaTech/CS7641/aelkugia3/aelkugia3/bill_authentication.csv")  

bankdata.head()  

# dividing data into attributes and labels

X = bankdata.drop('Class', axis=1)  # all columns except Class stored in X
y = bankdata['Class'] # only Class column stored in y


stats = []
stats2 = []
x = []

for i in np.arange(0.2, 1.0, 0.2):  
# dividing data into training and test sets

	from sklearn.model_selection import train_test_split  
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i)  

# training the algorithm using Scikit-Learn SVM library
# SVC = support vector classifier class

	from sklearn.svm import SVC  
	svclassifier = SVC(kernel='linear')  
	svclassifier.fit(X_train, y_train)

	svclassifier2 = SVC(kernel='rbf') 

	svclassifier2.fit(X_train, y_train)


# making predictions

	y_pred = svclassifier.predict(X_test) 

	start = time.clock()  
	y_pred2 = svclassifier2.predict(X_test) 
	print time.clock()

# Evaluating the algorithm (metrics including confusion matrix, precision, recall, and F1 score)

	from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
	# print(confusion_matrix(y_train,y_pred))  
	# print(classification_report(y_train,y_pred)) 

	print(confusion_matrix(y_test,y_pred2))  
	print(classification_report(y_test,y_pred2)) 

	stats.append(recall_score(y_test, y_pred))

	stats2.append(recall_score(y_test, y_pred2))

	x.append((1-i)*100)

plt.plot(x, stats)
plt.title("Training Size vs. Recall Score (Kernel = Linear)")
plt.xlabel("Training Size (%)")
plt.ylabel("Recall Score")
plt.show()

plt.plot(x, stats2)
plt.title("Training Size vs. Recall Score (Kernel = RBF)")
plt.xlabel("Training Size (%)")
plt.ylabel("Recall Score")
plt.show()





