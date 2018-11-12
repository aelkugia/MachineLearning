Project Title: Supervised Learning

Credits:

For algorithms -

https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

For Data -

Breast Cancer (https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival)
Bill Authentication (https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

How to Use/Tests:

1) Prior to running the code - update the pd.read_csv line to list the correct location of where the data is

2) Currently test size is listed as i, iteratively being captured. This number can be updated to reflect the train/test split. 

3) For each of the algorithms, plots are appearing showing the training size vs recall score

4) Timing of the predictions or training can be identified by using the time.clock() call 

Other notes:

Ensure the following is installed - numpy, matplotlib, pandads, time

for breast_cancer 

Survival Status changed to Class

1 survives --> 0
2 does not survive --> 1

To set up prediction on training, make the following instance updates:

From: X_test 
To: X_train

From: y_test
To: y_train

Example:

y_pred = classifier.predict(X_train)  
print(confusion_matrix(y_train, y_pred))  
print(classification_report(y_train, y_pred))  
stats.append(recall_score(y_train, y_pred))