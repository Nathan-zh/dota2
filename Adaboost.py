from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
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
names = ['Decision Tree', 'Nearest Neighbors', 'SVM', 'Neural Net', 'Naive Bayes']
classifiers = [DecisionTreeClassifier(max_depth=5), KNeighborsClassifier(3), SVC(),
               MLPClassifier(alpha=0.55), GaussianNB()]
clfs = [] # bad learners
y_pres = [] # collection of train predictions
y_test_pres = [] # collection of test predictions

for name, classifier in zip(names, classifiers):
    classifier.fit(array_train[:, 4:], array_train[:, 0])
    clfs.append(classifier)
    y_pres.append(classifier.predict(array_train[:, 4:]))
    acc_train = score(array_train[:, 0], y_pres[-1])

    y_test_pres.append(classifier.predict(array_test[:, 4:]))
    acc = score(array_test[:, 0], y_test_pres[-1])
    print('The accuracy of model %s is %.4f / %.4f' % (name, acc_train, acc))

# Adaboost
print('****************Adaboost start off!****************')
num = array_train.shape[0]
weights = np.ones(num) / num
losses = []

for i in range(len(names)):
    aa = y_pres[i] - array_train[:, 0]
    loss = np.dot(weights, np.multiply(aa, aa))
    losses.append(loss)

min_index = np.argmin(losses)
print('****************Update precedure****************')
iteration = 1000
w = []
index = [min_index]
for iters in range(iteration):

    epsilon = np.dot(weights, np.abs(y_pres[index[-1]] - array_train[:, 0]) / 2)
    w.append(0.5 * np.log(1 / epsilon - 1) + 0.3)
    if w[-1] < 0.01:
        break
    temp = np.multiply(weights, np.exp(-w[-1] * np.multiply(y_pres[index[-1]], array_train[:, 0])))
    weights = temp / np.sum(temp)
    losses = []

    for i in range(len(names)):
        aa = y_pres[i] - array_train[:, 0]
        loss = np.dot(weights, np.multiply(aa, aa))
        losses.append(loss)

    index.append(np.argmin(losses))
    print('Iteration = ', iters + 1, '  Index = ', index[-2], '  w = {:.4f}'.format(w[-1]))

h_b = np.zeros(array_test[:, 0].shape)

for k in range(len(index)-1):
    h_b += w[k] * y_test_pres[index[k]]

h_b = np.sign(h_b)
acc = score(array_test[:, 0], h_b)
print('****************The accuracy of ensemble model is %.4f****************' % acc)


