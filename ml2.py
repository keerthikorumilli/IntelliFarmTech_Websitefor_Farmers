import numpy as np
import pandas as pd
from operator import itemgetter

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
import pickle

warnings.filterwarnings("ignore")

# Replace the file name with your dataset file
df = pd.read_csv("Soil.csv")

# Update column names
X = df[['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']]
y = df['Output']

X = np.array(X)
y = np.array(y)

grid = {
    "n_estimators": [i for i in range(200, 1200, 10)],
    "max_depth": [i for i in range(1, 30)],
    "max_features": ["auto", "sqrt"],
    "min_samples_split": [i for i in range(2, 7)],
    "min_samples_leaf": [i for i in range(1, 6)]
}

clf = RandomForestClassifier(n_jobs=1)

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=10,  # number of models to try
                            cv=5,
                            verbose=2)

# Fit the RandomizedSearchCV version of clf
rs_clf.fit(X, y)
params = rs_clf.best_params_

n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth = itemgetter(
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "max_depth"
)(params)

# Initializing our model with the best parameters
# Parameters were found out by using RandomizedSearchCV
rfc = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, max_features=max_features,
                            max_depth=max_depth)
rfc.fit(X, y)
pickle.dump(rfc, open('model2.pkl', 'wb'))



