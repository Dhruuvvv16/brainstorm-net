import numpy as np
from Neural_Network import DeepNN
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

d = load_wine()
X = d.data
y = d.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = y_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train)
Y = y_train_onehot.T
model = DeepNN(X_train,Y,'categorical_crossentropy',alpha=0.01,lambda_=0.01)
model.train(epochs=1000)
predictions = model.predict(X_test)
pred = np.squeeze(predictions)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))