#loading required packages
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#loading data
iris = datasets.load_iris()

#adding features and labels
features = iris.data
labels = iris.target

#training and fitting the model
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[31, 1, 15, 10]])

print(preds)

plt.plot(features, labels)
plt.show()
