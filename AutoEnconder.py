import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score as score
#import numpy as np
from keras.models import Model
from keras.layers import Dense, Input

TRAIN_DATA_FILE = 'dota2Train.csv'
TEST_DATA_FILE = 'dota2Test.csv'
train = pd.read_csv(TRAIN_DATA_FILE, header=None)
test = pd.read_csv(TEST_DATA_FILE, header=None)
array_train = train.values
array_test = test.values
x_train = array_train[:, 4:]

input_img = Input(shape=(113,))
encoded1 = Dense(64, activation='tanh')(input_img)
encoded2 = Dense(32, activation='tanh')(encoded1)

decoded2 = Dense(32, activation='tanh')(encoded2)
decoded3 = Dense(113, activation='tanh')(decoded2)

autoencoder = Model(inputs=input_img, outputs=decoded3)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_split=0.1)

enconder = Model(inputs=input_img, outputs=encoded2)
x_train_rep = enconder.predict(x_train)
print(x_train_rep[:10, :])

classifier = svm.SVC()
classifier.fit(x_train_rep, array_train[:, 0])

x_test_rep = enconder.predict(array_test[:, 4:])
y_pre = classifier.predict(x_test_rep)

acc = score(array_test[:, 0], y_pre)
print(acc)
