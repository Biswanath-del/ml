from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

Breast_cancer = datasets.load_breast_cancer()

Breast_cancer_x = Breast_cancer.data[:, np.newaxis, 4]

Breast_cancer_x_train = Breast_cancer_x[-30:]
Breast_cancer_x_test = Breast_cancer_x[:-30]

Breast_cancer_y_train = Breast_cancer.target[-30:]
Breast_cancer_y_test = Breast_cancer.target[:-30]

clf = KNeighborsClassifier()
clf.fit(Breast_cancer_x_train, Breast_cancer_y_train)

predes = clf.predict(Breast_cancer_x_test)
print(predes)

plt.scatter(Breast_cancer_x_test, Breast_cancer_y_test)
plt.plot(Breast_cancer_x_test, predes)
plt.show()





