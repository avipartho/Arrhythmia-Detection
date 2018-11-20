from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from essentials import plot_confusion_matrix,corr_heatmap
import pandas as pd

# loading data
y = np.loadtxt('y.csv')
x = np.loadtxt('Data.csv',delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23))
# x = np.loadtxt('Data.csv',delimiter=',')

# corr_heatmap(pd.read_csv('Data.csv'))

# Feature normalization
mu, std, x_normalized = np.mean(x, axis = 0), np.std(x, axis = 0), []
for i,j,k in zip(x.T,mu,std):
	x_normalized.append((i-j)/k)
x = np.asarray(x_normalized).T

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5) # 20% as test
class_names = ['Normal','P','LBBB','RBBB','PVC'] # 5 classes

#SVM
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_train, y_train.ravel())
pred = clf.predict(x_test)
print("Accuracy from svm: {:g}".format(accuracy_score(y_test.ravel(), pred.ravel()))) 
# print(clf.support_vectors_)

cnf_matrix =  confusion_matrix(y_test.ravel(), pred.ravel())
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')


#SVM using stochastic gradient descent
lr = SGDClassifier(loss="hinge", penalty="l2", alpha = 0.001, max_iter=1000)
lr.fit(x_train, y_train.ravel())
pred = lr.predict(x_test)
print("Accuracy from svm: {:g}".format(accuracy_score(y_test.ravel(), pred.ravel())))

cnf_matrix =  confusion_matrix(y_test.ravel(), pred.ravel())
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
# for xs, ys in minibatches:
#     lr.partial_fit(xs, ys, classes=classes)

#Random forest
clf = RandomForestClassifier(random_state=0)
clf.fit(x_train, y_train.ravel())
pred = clf.predict(x_test)
print("Accuracy from Rf: {:g}".format(accuracy_score(y_test.ravel(), pred.ravel()))) 
# print(clf.support_vectors_)

cnf_matrix =  confusion_matrix(y_test.ravel(), pred.ravel())
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')