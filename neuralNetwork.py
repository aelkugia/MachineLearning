#CS7641: Machine Learning
#Fall 2018
#aelkugia3

#Neural Network
#Citation: https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

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

#Create training and test splits - 80% training set, 20% test data

stats = []
x = []

for i in np.arange(0.2, 1.0, 0.2):

	from sklearn.model_selection import train_test_split  

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= i) 

#Scaling the features so that all can be uniformly evaluated

	from sklearn.preprocessing import StandardScaler 
	scaler = StandardScaler()  
	scaler.fit(X_train)

	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)  

	#Training the NN

	from sklearn.neural_network import MLPClassifier 
	mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000) #hidden layers set to 3 layers of 10 nodes each, 1000 iterations
	
	mlp.fit(X_train, y_train.values.ravel()) #train algorithm to training data (time this, printing iteration)

	predictions = mlp.predict(X_train) #predictions on our test data time (time this)

# Evaluating the algorithm (metrics including confusion matrix, precision, recall, and F1 score)

	from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
	print(confusion_matrix(y_train,predictions))  
	print(classification_report(y_train,predictions))   

	stats.append(recall_score(y_train, predictions))

	x.append((1-i)*100)

plt.plot(x, stats)
plt.title("Training Size vs. Recall Score")
plt.xlabel("Training Size (%)")
plt.ylabel("Recall Score")
plt.show()


