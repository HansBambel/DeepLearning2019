import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.head()
df = df[:100]
classes = df[4].unique()
samples = df.copy()
samples[4][samples[4] == classes[0]] = 0
samples[4][samples[4] == classes[1]] = 1
samples = preprocessing.MinMaxScaler().fit_transform(samples)
# print(samples)
featuresToDelete = [1, 3]
data = np.delete(samples, featuresToDelete, axis=1)
data = np.random.permutation(data)
data_train = data[:int(0.8*len(data))]
# print(data_train)
data_test = data[int(0.8*len(data)):]
samples = data[:, :2]
targets = data[:, 2]
print(targets.shape)
print(samples.shape)
# print(samples)
# print(samples.shape)
# print(targets)
# print(targets.shape)
weights = np.random.randn(3)
lr = 0.01
# print(w)
lossTrace = []

def forwardPass(x):
    inp = x
    inp = np.insert(inp, 0, 1, axis=0)
    a = np.sum(weights * inp)
    sigmoidal = 1 / (1 + math.exp(-a))
    return sigmoidal

for i in range(3000):

    for j in range(len(samples)):
        out = forwardPass(samples[j])
        inp2 = np.insert(samples[j], 0, 1, axis=0)
        # if j < 50: print(a), print(out)
        for r in range(weights.shape[0]):
            gradient = (targets[j] - out) * inp2[r]
            weights[r] -= lr * -gradient

        prediction = 1 - 1e-15 if out == 1 else out
        prediction = 1e-15 if out == 0 else out
        loss = -targets[j] * math.log(prediction) - (1 - targets[j]) * math.log(1 - prediction)
        lossTrace.append(loss)
print(loss)
test = data_test[:, :2]
testTarget = data_test[:, 2]
for q in range(len(test)):
    print(f'Target: {testTarget[q]} prediction: {forwardPass(test[q])}')