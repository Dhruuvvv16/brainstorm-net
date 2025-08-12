import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
Y = data.target
X = np.c_[np.ones((X.shape[0], 1)), X]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)