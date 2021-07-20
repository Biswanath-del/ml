import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = load_breast_cancer()
df = pd.DataFrame(np.c_[data.data, data.target], columns=[list(data.feature_names)+['target']])

x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2020)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)


predict = clf.predict(x_test)
print(predict)

plt.scatter(x_test, y_test)
plt.plot(x_test, predict)
plt.show()



