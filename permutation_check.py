from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt


import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data

y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train, y_train)

results = permutation_importance(rf, x_test, y_test, n_repeats=10, random_state = 42)

import_df = pd.DataFrame({
    'features':iris.feature_names,
    'Importance': results.importances_mean
}).sort_values(by='Importance')

plt.barh(import_df['features'],import_df['Importance'])
plt.show()

