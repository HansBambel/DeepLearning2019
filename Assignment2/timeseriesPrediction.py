import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
import statsmodels
import keras
from keras.layers import LSTM, Dense, Activation

data = sio.loadmat("Xtrain.mat")["Xtrain"]
data_normed = preprocessing.minmax_scale(data)

print(data.shape)
# plt.subplot(121)
# plt.plot(data)
# plt.title("Data")
# plt.xlabel("Time step")
#
# plt.subplot(122)
# plt.plot(data_normed)
# plt.title("Data (normalized)")
# plt.xlabel("Time step")
# plt.show()

# TODO use ensemble
# Test is length 200
# Input is window of 50 points
inputsize = 100

model = keras.Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(inputsize, 1), dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# Maybe this does not need to be done (just use normal data)
# Create train and test data (model is trained on sequence and should predict the next "window" timesteps
targetIndex = np.array(range(inputsize, len(data)))
# print(targetIndex-100)
inputdata = []
target = []
for ind in targetIndex:
    inputdata.append(data[ind-inputsize:ind])
    target.append(ind)
inputdata = np.array(inputdata)
target = np.array(target)
# inputdata = data[targetIndex-inputsize:targetIndex+window]
print(inputdata.shape)

xData = inputdata[:, :100]
print(f'xData Shape {xData.shape}')
print(f'target Shape {target.shape}')

# Split into validation
splitInd = int(len(xData)*0.9)
xData_train = xData[:splitInd]
target_train = target[:splitInd]
xData_val = xData[splitInd:]
target_val = target[splitInd:]
print("xData_train.shape", xData_train.shape)
print("target_train.shape", target_train.shape)
print("xData_val.shape", xData_val.shape)
print("target_val.shape", target_val.shape)

model.summary()
epochs = 1
history = model.fit(xData_train, target_train, epochs=epochs, shuffle=True, validation_data=(xData_val, target_val))

pred = model.predict(xData_val)
print("pred.shape ", pred.shape)
plt.plot(pred, 'o', c='r')
plt.plot(target_val, 'o', c='g')
plt.show()