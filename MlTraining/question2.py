import numpy as np
import pandas as pd
import sys

sys.path.append("/Users/fukadakengo/dev/")
import rakus_ml_training as rmt


def load_data():
    train_data = rmt.cancer.get_train_data()
    test_data = rmt.cancer.get_test_data()
    return train_data, test_data


def add_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def sigmoid(w, X):
    return 1 / (1 + np.exp(-(np.dot(X, w.T))))


def cross_entropy(w, X, t):
    return np.mean(- (t * np.log(sigmoid(w, X)) + (1 - t) * np.log(1 - sigmoid(w, X))))


def dcross_entropy(w, X, t):
    y = sigmoid(w, X)
    return np.dot(X.T, (y - t)) / X.shape[0]


def fit(w, X, t):
    max_iteration = 100000
    alpha = 0.03
    for i in range(max_iteration):
        w = w - alpha * dcross_entropy(w, X, t)
        print(f'iteration: {i},  error: {cross_entropy(w, X, t)}')
    return w


train, test = load_data()
train_x = np.array(train.drop('target', axis=1))
train_t = np.array(train.loc[:, 'target'])

mue = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x - mue) / std

train_x = add_bias(train_x)

w_init = np.zeros(train_x.shape[1])
w = fit(w_init, train_x, train_t)
print(w)

train_predict = sigmoid(w, train_x)
rmt.cancer.confirm(pd.DataFrame(train_predict), pd.DataFrame(train_t))

test = (test - mue) / std

test = add_bias(test)

predict = sigmoid(w, test)
rmt.cancer.confirm(pd.DataFrame(predict))
