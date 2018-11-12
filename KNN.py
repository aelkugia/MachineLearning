#CS7641: Machine Learning
#Fall 2018
#aelkugia3

#k-nearest neighbors
#Citation: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import time

dataset = pd.read_csv("/Users/aelkugia/Documents/GeorgiaTech/CS7641/aelkugia3/aelkugia3/bill_authentication.csv")  #Update with dataset

dataset.shape #identify number of rows/columns in dataset
dataset.head() #inspect the first five records of the dataset

# Dividing data into attributes and labels
# X -> attribute set
# y -> corresponding labels

X = dataset.drop('Class', axis=1) # X variable contains all columns from the dataset, except "Class"
y = dataset['Class'] # Y variable contains the values from the "Class Column"  

stats = []
x = []


# dividing data into training and test sets

for i in np.arange(0.2, 1.0, 0.2):

	from sklearn.model_selection import train_test_split  
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i) 

	# Scaling features so that all of them can be uniformally evaluated

	from sklearn.preprocessing import StandardScaler  
	scaler = StandardScaler()  
	scaler.fit(X_train)

	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)  

	# Training the KNN algorithm

	from sklearn.neighbors import KNeighborsClassifier 

	from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

	classifier = KNeighborsClassifier(n_neighbors=14)  


	classifier.fit(X_train, y_train)  


	# Predictions for KNN 
	start = time.clock() 
	y_pred = classifier.predict(X_train)  
	print time.clock()

	# Evaluating the algorithm (metrics including confusion matrix, precision, recall, and F1 score)

	print(confusion_matrix(y_train, y_pred))  
	print(classification_report(y_train, y_pred)) 

	stats.append(recall_score(y_train, y_pred))

	x.append((1-i)*100)

plt.plot(x, stats)
plt.title("Training Size vs. Recall Score")
plt.xlabel("Training Size (%)")
plt.ylabel("Recall Score")
plt.show()


### Identifying the ideal number of neighbors

# x = []
# stats = []

# for i in range(3,20):

# 	classifier = KNeighborsClassifier(n_neighbors=i)  
# 	classifier.fit(X_train, y_train)  

# # Predictions for KNN 

# 	y_pred = classifier.predict(X_test)  

# # Evaluating the algorithm (metrics including confusion matrix, precision, recall, and F1 score)

# 	print(confusion_matrix(y_test, y_pred))  
# 	print(classification_report(y_test, y_pred))  

# 	#####TODO prediction on training####

# 	# y_pred2 = classifier.predict(X_train)  

# 	# print(confusion_matrix(y_train, y_pred2))  
# 	# print(classification_report(y_train, y_pred2)) 

# 	stats.append(recall_score(y_test, y_pred))

# 	x.append(i)

# plt.plot(x, stats)
# plt.title("K-Neighbors vs. Recall Score")
# plt.xlabel("# of Neighbors")
# plt.ylabel("Recall Score")
# plt.show()





