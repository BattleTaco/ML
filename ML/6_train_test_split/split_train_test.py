# split_train_test.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('carprices.csv')

plt.scatter(df.Mileage, df['Sell Price($)'])
#plt.scatter(df['Age(yrs)'], df['Sell Price($)'])

X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train, y_train)
print(clf.predict(X_test))
print()
print(y_test)

print(clf.score(X_test, y_test))