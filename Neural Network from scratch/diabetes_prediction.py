import numpy as np
from Neural_Network import DeepNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
d = load_diabetes()
X = d.data
Y = d.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_train_mean) / X_train_std
model = DeepNN(X_train,y_train,'mse',alpha=0.01)
model.train(epochs=1000)
predictions = model.predict(X_test)
pred = np.squeeze(predictions)
print(r2_score(y_test,pred))
