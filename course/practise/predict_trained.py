from keras.models import load_model

import numpy as np

model = load_model('cancer.h5')

data = np.genfromtxt('cancer_data_set.csv', delimiter=',')

x_predict = data[0:, 1:]

output = model.predict_classes(x_predict)

print(output)