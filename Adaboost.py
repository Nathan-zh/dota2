from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score as score

# Data
TRAIN_DATA_FILE = 'dota2Train.csv'
TEST_DATA_FILE = 'dota2Test.csv'

train = pd.read_csv(TRAIN_DATA_FILE, header=None)
test = pd.read_csv(TEST_DATA_FILE, header=None)

# y = array_train[:,0], x = array_train[:, 4:]
# size: y: n*1, x: n*113
array_train = train.values
array_test = test.values

# Single models
names = ["Nearest Neighbors", "SVM", "Decision Tree", "Neural Net", "Naive Bayes"]
classifiers = [KNeighborsClassifier(3), SVC(), DecisionTreeClassifier(max_depth=5),
               MLPClassifier(alpha=0.55), GaussianNB()]

for name, classifier in zip(names, classifiers):
    classifier.fit(array_train[:, 4:], array_train[:, 0])
    y_pre = classifier.predict(array_test[:, 4:])
    acc = score(array_test[:, 0], y_pre)
    print('The accuracy of model %s is %.4f' % (name, acc))

# Adaboost


