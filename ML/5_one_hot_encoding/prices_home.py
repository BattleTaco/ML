# price_home.py
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')

#creating dummy variables to help read text

dummies = pd.get_dummies(df.town) # grabs all text files and convertes the text into columns and replaces values with 1 or 0

merged = pd.concat([df, dummies], axis = 'columns')


final_merge = merged.drop(['town', 'west windsor'], axis = 'columns')
# print(final_merge)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

X = final_merge.drop('price', axis = 'columns')
# print(X)

y = final_merge.price

reg.fit(X,y)

# print('Prediction:',reg.predict([[2800, 0, 1]]))
# print('Prediction 2:', reg.predict([[3400, 0, 0]]))

# print(reg.score(X,y))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df
dfle.town = le.fit_transform(dfle.town)


X = dfle[['town', 'area']].values
y = dfle.price

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [0])

X = ohe.fit_transform(X).toarray()
print(X)