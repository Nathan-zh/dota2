import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score as score
from sklearn.decomposition import PCA

TRAIN_DATA_FILE = 'dota2Train.csv'
TEST_DATA_FILE = 'dota2Test.csv'

train = pd.read_csv(TRAIN_DATA_FILE, header=None)
test = pd.read_csv(TEST_DATA_FILE, header=None)

# y = array_train[:,0], x = array_train[:, 4:]
# size: y: n*1  x: n*113
array_train = train.values
array_test = test.values

train_x = array_train[:, 4:]
test_x = array_test[:, 4:]

pca = PCA(n_components=64)
train_z = pca.fit_transform(train_x)
test_z = pca.transform(test_x)

classifier = svm.SVC()
classifier.fit(train_z, array_train[:, 0])
y_pre = classifier.predict(test_z)

acc = score(array_test[:, 0], y_pre)
print(acc)
