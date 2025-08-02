import numpy as np
from Neural_Network import DeepNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = load_breast_cancer()
X = data.data
Y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = DeepNN(X_train,y_train,'binary_crossentropy',alpha=0.01)
model.train(epochs=1000)
predictions = model.predict(X_test)
predictions = np.squeeze(predictions)
print(classification_report(y_test,predictions))
