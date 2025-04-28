from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
iris = load_iris()

x = iris.data

y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

os.makedirs('model', exist_ok=True)
model_path = os.path.join('model', 'model.joblib')
joblib.dump(knn, model_path)


