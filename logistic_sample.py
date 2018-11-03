from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

sorted_df = df.sort_values(by='target')

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

clf = LogisticRegression()
clf.fit(X_train, y_train)

print(clf.coef_, clf.intercept_)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
