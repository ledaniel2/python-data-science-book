import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data
y = iris.target

lr = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(lr, X, y, cv=5)
mean_cv_score = np.mean(cv_scores)
print("Average Cross-Validation Score: {:.2f}".format(mean_cv_score))
