import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import keras
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import EarlyStopping

data = sio.loadmat("Xtrain.mat")["Xtrain"]
# create a scaler and fit it on data
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data)

data_normed = scaler.transform(data)

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

inputsize = 100

def evaluate_model(inp, target, epochs=1):
    # encode targets

    # define model
    model = keras.Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(inp.shape[1], 1), dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    # fit model
    # model.fit(trainX, trainy_enc, epochs=50, verbose=0)
#     early_stopping_monitor = EarlyStopping(monitor='loss', patience=3)
    history = model.fit(inp, target, epochs=epochs, validation_split=0.1, verbose=2)
    # evaluate the model
    # _, test_acc = model.evaluate(testX, testY, verbose=0)
    return model, history


def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    # TODO maybe mean?
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result


# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)



# model = keras.Sequential()
# model.add(LSTM(32, return_sequences=True, input_shape=(inputsize, 1), dropout=0.2, recurrent_dropout=0.2))
# model.add(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='relu'))
#
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()

# TODO use ensemble
# Test is length 200

# Create train and test data (model is trained on sequence and should predict the next timestep)
targetIndex = np.array(range(inputsize, len(data)))
# print(targetIndex-100)
inputdata = np.array([data_normed[ind-inputsize:ind] for ind in targetIndex])
target = data_normed[targetIndex]

print(f'inputdata Shape {inputdata.shape}')
print(f'target Shape {target.shape}')

# Split into train and test
splitInd = int(len(inputdata)*0.9)
trainX = inputdata[:splitInd]
trainY = target[:splitInd]
testX = inputdata[splitInd:]
testY = target[splitInd:]
print("trainX.shape", trainX.shape)
print("trainY.shape", trainY.shape)
print("testX.shape", testX.shape)
print("testY.shape", testY.shape)

# model, history = evaluate_model(trainX, trainY, testX, testY, epochs=5)
model, history = evaluate_model(trainX, trainY, epochs=50)

# plt.plot(history.history['loss'])
# plt.show()
pred = model.predict(testX)
pred_unscaled = scaler.inverse_transform(pred)
testY_unscaled = scaler.inverse_transform(testY)
print("pred.shape ", pred.shape)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title('Normalized prediction')
plt.plot(pred, 'o', c='r', label="Predicted")
plt.plot(testY, 'o', c='g', label="Target")
plt.legend()

plt.subplot(122)
plt.title('Unscaled prediction')
plt.plot(pred_unscaled, 'o', c='r', label="Predicted")
plt.plot(testY_unscaled, 'o', c='g', label="Target")
plt.legend()
plt.show()

plt.title('Unscaled prediction')
plt.plot(pred_unscaled, 'o', c='r', label="Predicted")
plt.plot(testY_unscaled, 'o', c='g', label="Target")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()