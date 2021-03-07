# insurance.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('insurance_data.csv')

plt.scatter(df.age, df.bought_insurance, marker = '*', color = 'orange')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size = 0.9) #x must be a multi dimensional array to work

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

print(X_test)
print('Prediction:',model.predict(X_test))

print('Accuracy:', model.score(X_test, y_test))
