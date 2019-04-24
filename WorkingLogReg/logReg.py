import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing

def forwardPass(x):
    inp = np.insert(x, 0, 1, axis=0)
    a = np.dot(weights, inp)
    sigmoid = 1 / (1 + math.exp(-a))
    return sigmoid


def regTerm(i, lamReg, reg):
    if i != 0 & reg == True:
        return lamReg / 2 * np.sum(weights ** 2)
    return 0


def trainNN(regularization, learningRate, featuresToDelete, samples):

    eps = 1e-7
    data = np.delete(samples, featuresToDelete, axis=1)
    data = np.random.permutation(data)
    data_train = data[:int(0.8 * len(data))]
    # print(data_train)
    data_test = data[int(0.8 * len(data)):]
    samples = data_train[:, :2]
    targets = data_train[:, 2]
    epochs = 0
    converged = False
    while not converged:
        epochs += 1
        for j in range(len(samples)):
            out = forwardPass(samples[j])
            inp = np.insert(samples[j], 0, 1, axis=0)
            for r in range(len(weights)):
                gradient = (targets[j] - out) * inp[r]
                weights[r] -= learningRate * -gradient + regTerm(r, 0.001, regularization)

            pred = np.clip(out, eps, 1 - eps)
            loss = -targets[j] * math.log(pred) - (1 - targets[j]) * math.log(1 - pred)
        # if epochs%100 == 0:
        #     print(f'Error in epoch {epochs}: {loss}')
        if np.abs(loss) < 0.005:
            converged = True
            print(f'Converged after {epochs} epochs.')
    test = data_test[:, :2]
    testTarget = data_test[:, 2]
    test_error = 0
    for i in range(len(test)):
        test_error += (testTarget[i]-forwardPass(test[i]))**2
    print(f'Error on test set: {test_error}')
    # print(f'Target: {testTarget[q]} prediction: {forwardPass(test[q])}')


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.head()
df = df[:100]
classes = df[4].unique()
samples = df.copy()
samples[4][samples[4] == classes[0]] = 0
samples[4][samples[4] == classes[1]] = 1
samples = preprocessing.MinMaxScaler().fit_transform(samples)
samples2 = samples.copy()
weights = np.random.randn(3)
weights2 = weights.copy()
trainNN(True, 0.1, [1, 3], samples)


def plotConverge(samples, features):
    step_size = .01
    x_min, x_max = samples[:, features[0]].min() - 1, samples[:, features[0]].max() + 1
    y_min, y_max = samples[:, features[1]].min() - 1, samples[:, features[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                            np.arange(y_min, y_max, step_size))
    Z = np.array(list(map(forwardPass, np.c_[xx.ravel(), yy.ravel()])))

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap="hot", levels=1)

    # plot data points
    plt.scatter(x=samples[features[0]], y=samples[features[1]], cmap="tab10")

    # plt.title(f"Features {stats[2][0]} and {stats[2][1]}; Accuracy: {stats[0]}, Loss: {'%.2E' % stats[1]}")


    plt.show()

#plotConverge(samples, [0, 2])