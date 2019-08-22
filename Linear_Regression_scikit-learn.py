import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

np.random.seed(0)
x = np.array([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
y = np.array([0.5,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5])

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()