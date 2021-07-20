from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)

y_predict = knn.predict([[4, 6, 9, 56]])
print(y_predict)


