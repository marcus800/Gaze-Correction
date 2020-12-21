from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.random.rand(100,2)
y=X[:,0]**2-3*X[:,1]**3+2*X[:,0]*X[:,1]-5

print(X.shape)
print(y.shape)
poly = PolynomialFeatures(degree=3)
X_t = poly.fit_transform(X)
print(X_t.shape)


clf = LinearRegression()
clf.fit(X_t, y)
print(clf.coef_)
print(clf.intercept_)