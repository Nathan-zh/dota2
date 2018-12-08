from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
from sklearn.metrics import accuracy_score as score

TRAIN_DATA_FILE = 'dota2Train.csv'
TEST_DATA_FILE = 'dota2Test.csv'

train = pd.read_csv(TRAIN_DATA_FILE, header=None)
test = pd.read_csv(TEST_DATA_FILE, header=None)

# y = array_train[:,0], x = array_train[:, 4:]
# size: y: n*1  x: n*113
array_train = train.values
array_test = test.values

classifier1 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
classifier1.fit(array_train[:, 4:], array_train[:, 0])
y_pre1 = classifier1.predict(array_test[:, 4:])
acc1 = score(array_test[:, 0], y_pre1)
print(acc1)

classifier2 = AdaBoostClassifier()
classifier2.fit(array_train[:, 4:], array_train[:, 0])
y_pre2 = classifier2.predict(array_test[:, 4:])
acc2 = score(array_test[:, 0], y_pre2)
print(acc2)
