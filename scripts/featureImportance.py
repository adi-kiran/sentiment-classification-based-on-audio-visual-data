#%%
l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

req = generateColumns(1, 68)

import pandas as pd
df = pd.read_csv('merge-mix.csv')

# selecting features and label as X & y respectively
X = df[req]
y = df['emotion']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

from sklearn.ensemble import GradientBoostingRegressor as GB
gb = GB()
gb.fit(X_train, y_train.values.ravel())

import matplotlib.pyplot as plt
plt.bar(range(X_train.shape[1]), gb.feature_importances_)
plt.xticks(range(X_train.shape[1]), req)
plt.show()
