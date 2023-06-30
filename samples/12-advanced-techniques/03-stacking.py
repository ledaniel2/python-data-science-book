from sklearn.datasets import load_iris
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(random_state=42)),
]

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
)
clf.fit(X, y)

y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Stacking accuracy: ", accuracy)
