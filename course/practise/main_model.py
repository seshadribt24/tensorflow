import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

from keras.optimizers import Adam


model = Sequential()

model.add(Dense(8, activation='relu', input_dim=30))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

opt = Adam(lr=0.001)

model.compile(
  optimizer=opt, 
  loss='binary_crossentropy',
  metrics=['accuracy']
)

data = np.genfromtxt('cancer_data_set.csv', delimiter=',')

x_train = data[0:, 1:]
y_train = to_categorical(data[0:, 0])


perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

model.fit(
  x_train, 
  y_train,
  epochs=100, 
  validation_split=0.2
)

model.optimizer = Adam(lr=0.0001)

model.fit(
  x_train, 
  y_train,
  epochs=100, 
  validation_split=0.2
)

model.save('cancer.h5')