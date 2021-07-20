from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

diabates = datasets.load_diabetes()

diabates_x = diabates.data[:, np.newaxis, 2]

diabtes_x_train = diabates_x[:-30]
diabates_x_test = diabates_x[-30:]

diabates_y_train = diabates.target[:-30]
diabates_y_test = diabates.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabtes_x_train, diabates_y_train)

diabates_y_predicted = model.predict(diabates_x_test)

print("Mean square error is : ", mean_squared_error(diabates_y_test, diabates_y_predicted))

print("weights: ", model.coef_)
print("intercept: ", model.intercept_)

plt.scatter(diabates_x_test, diabates_y_test)
plt.plot(diabates_x_test, diabates_y_predicted)

plt.show()




