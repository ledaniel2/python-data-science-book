from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

print('Independent Variable(s): X')
print(X[0:5])
print('Dependent Variable: y')
print(y)
