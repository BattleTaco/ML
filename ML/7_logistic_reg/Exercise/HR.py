# HR.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('HR_comma_sep.csv')

plt.bar(df.salary, df.left)
plt.bar(df.Department, df.left)

X = df[['number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'satisfaction_level', 'last_evaluation']]
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9)

model = LogisticRegression()
model.fit(X_train, y_train)

print('accuracy:', model.score(X_test, y_test))