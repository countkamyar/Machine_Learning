import operator
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
x = np.linspace(0,1,50)
y = np.cos(x)+np.random.uniform(-0.5, 0.5, 50)

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]
polynomialfeatures=PolynomialFeatures(degree=2)
xpoly=polynomialfeatures.fit_transform(x)
model = LinearRegression()
model.fit(xpoly, y)
y_pred_poly = model.predict(xpoly)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_pred_poly), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)

plt.scatter(x, y, s=10)
plt.plot(x, y_pred_poly, color='r')
plt.show()