from keras import Sequential, backend as K
from keras.layers import Dense, multiply, Dropout
import numpy
from keras.optimizers import RMSprop
from numpy import loadtxt as np_loadtxt, array as np_array

from src.data import getTrain, getTest

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model = Sequential()
model.add(Dense(32, kernel_initializer='normal', activation="relu", input_shape=(3,)))
model.add(Dense(64, kernel_initializer='normal', activation="relu"))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss=root_mean_squared_error,
              optimizer="adam",
              metrics=['accuracy'])

print(model.summary())

results = np_loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
x_train = []
y_train = []
for result in results:
    x_train.append(result[0:3])
    y_train.append(result[3])
x_train = np_array(x_train)
y_train = np_array(y_train)

results = np_loadtxt(open("predict.csv", "rb"), delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
x_test = []
y_test = []
for result in results:
    x_test.append(result[0:3])
    y_test.append(result[3])
x_test = np_array(x_test)
y_test = np_array(y_test)

history = model.fit(x_train, y_train,
                    batch_size=124,
                    epochs=15,
                    verbose=1,
                    validation_data=(x_test, y_test))

classes = model.predict(x_test, batch_size=128)
with open("jhk-prediction.txt", "w") as f:
    for class_ in list(classes):
        f.write("{:.8g}\n".format(class_[0]))

