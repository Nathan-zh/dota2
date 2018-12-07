import pandas as pd
#from sklearn import svm
from sklearn.metrics import accuracy_score as score
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input

TRAIN_DATA_FILE = 'dota2Train.csv'
TEST_DATA_FILE = 'dota2Test.csv'
train = pd.read_csv(TRAIN_DATA_FILE, header=None)
test = pd.read_csv(TEST_DATA_FILE, header=None)
array_train = train.values
array_test = test.values

input_img = Input(shape=(113,))
x = Dense(64, activation='sigmoid')(input_img)
x = Dense(32, activation='sigmoid')(x)
x = Dense(16, activation='sigmoid')(x)
x = Dense(1, activation='tanh')(x)

embedding = Model(inputs=input_img, outputs=x)
embedding.compile(optimizer='adadelta', loss='binary_crossentropy')

embedding.fit(array_train[:, 4:], array_train[:, 0],
                epochs=50,
                batch_size=1024,
                shuffle=True,
                validation_split=0.1)

y_pre = embedding.predict(array_test[:, 4:])
print(y_pre[:10])
acc = score(array_test[:, 0], np.sign(y_pre))
print(acc)
