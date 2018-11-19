import numpy as np
import pandas as pd
import sys

sys.path.append("/Users/fukadakengo/dev/")
import rakus_ml_training as rmt

train = rmt.boston.get_train_data()
test = rmt.boston.get_test_data()

train_x = np.array(train.drop('TARGET', axis=1))
train_t = np.array(train.loc[:, 'TARGET'])

train_x = np.hstack((train_x, train_x**2))

mue = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x - mue) / std

train_x = np.hstack((np.ones((train_x.shape[0], 1)), train_x))


def hypothesis(w, X):
    return np.dot(X, w.T)


def dcost_function(w, X, t):
    y = hypothesis(w, X)
    return 2 * np.dot((y - t), X) / X.shape[0]


def fit(w, X, t):
    rate = 0.06
    for i in range(80000):
        w = w - rate * dcost_function(w, X, t)
        print(f'iteration: {i},  error: {np.mean((hypothesis(w, X) - t) ** 2)}')
    return w


w_init = np.zeros(train_x.shape[1])
W = fit(w_init, train_x, train_t)
print(W)

test = np.hstack((test, test**2))
test = (test - mue) / std
test = np.hstack((np.ones((test.shape[0], 1)), test))

predict = pd.DataFrame(hypothesis(W, test))

rmt.boston.confirm(predict)
