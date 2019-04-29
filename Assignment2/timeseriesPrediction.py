import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
import statsmodels
import keras
from keras.layers import LSTM, Dense, Activation

data = sio.loadmat("Xtrain.mat")["Xtrain"]
data_normed = preprocessing.minmax_scale(data)

print(data.shape)
plt.subplot(121)
plt.plot(data)
plt.title("Data")
plt.xlabel("Time step")

plt.subplot(122)
plt.plot(data_normed)
plt.title("Data (normalized)")
plt.xlabel("Time step")
# plt.show()

# TODO use ensemble
# Test is length 200
# Input is window of 50 points
inputsize = 100
window = 50

model = keras.Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(16, return_sequences=True))
# model.add(LSTM(16, return_sequences=False))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam')

# Maybe this does not need to be done (just use normal data)
# Create train and test data (model is trained on sequence and should predict the next "window" timesteps
targetIndex = np.array(range(inputsize, len(data)-window))
# print(targetIndex-100)
inputdata = []
for ind in targetIndex:
    inputdata.append(data[ind-inputsize:ind+window])
inputdata = np.array(inputdata)#.squeeze()
# inputdata = data[targetIndex-inputsize:targetIndex+window]
print(inputdata.shape)

xData = inputdata[:, :100]
target = inputdata[:, -50:]
print(f'xData Shape {xData.shape}')
print(f'target Shape {target.shape}')

# Split into validation
splitInd = int(len(xData)*0.9)
xData_train = xData[:splitInd]
target_train = target[:splitInd]
xData_val = xData[splitInd:]
target_val = target[splitInd:]

model.summary()

model.fit(xData_train, target_train, validation_data=(xData_val, target_val))