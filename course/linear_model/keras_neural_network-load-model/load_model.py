from keras.models import load_model

import numpy as np

model = load_model('mean_network.h5')

# Predict
x_predict = np.array([
  [1.5, 2, 3.5, 4],
  [13, 11, 9, 14],
  [102, 98.5, 102.5, 100],
  [10.5, 5, 9.5, 12]
])

output = model.predict(x_predict)

print("")
print("Expected: 2.75, 11.75, 100.75, 9.25")
print("Actual: ", output)

