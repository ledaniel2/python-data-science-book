from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

new_data = [[5.0, 3.6, 1.3, 0.25]]
print(clf.predict(new_data))
