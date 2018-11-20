import numpy as np
import pandas as pd
import sys

sys.path.append("/Users/fukadakengo/dev/")
import rakus_ml_training as rmt


def load_data():
    train_data = rmt.iris.get_train_data()
    test_data = rmt.iris.get_test_data()
    return train_data, test_data


def add_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def softmax(W, X):
    A = np.exp(np.dot(X, W.T))
    u = np.sum(A, axis=1)
    U = np.vstack((u, u, u)).T
    return A / U


def cross_entropy(W, X, T):
    return - np.mean(np.sum(T * np.log(softmax(W, X)), axis=1))


def dcross_entropy(W, X, T):
    Y = softmax(W, X)
    return np.dot((Y - T).T, X) / X.shape[0]


def fit(W, X, T):
    max_iteration = 10000
    alpha = 10
    for i in range(max_iteration):
        W = W - alpha * dcross_entropy(W, X, T)
        print(f'iteration: {i},  error: {cross_entropy(W, X, T)}')
    return W


train, test = load_data()
train_X = np.array(train.drop('target', axis=1))
train_T = pd.get_dummies(train['target']).get_values()

mue = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)
train_X = (train_X - mue) / std

train_X = add_bias(train_X)

W_init = np.zeros((train_T.shape[1], train_X.shape[1]))

softmax(W_init, train_X)
W = fit(W_init, train_X, train_T)
print(W)

train_predict_prob = softmax(W, train_X)
train_predict = np.argmax(train_predict_prob, axis=1)
rmt.iris.confirm(pd.DataFrame(train_predict), pd.DataFrame(train['target']))

test = (test - mue) / std
test = add_bias(test)

predict_prob = softmax(W, test)
predict = np.argmax(predict_prob, axis=1)
rmt.iris.confirm(pd.DataFrame(predict))
