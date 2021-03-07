# prices_for_home.py

import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('homeprices.csv')

# take median of bedrooms and set it as the NaN value

median_bedrooms = math.floor(df.bedrooms.median()) #this grabs median of the column
#needed to make this as a accurate and integer count

df.bedrooms = df.bedrooms.fillna(median_bedrooms)# fills all NA values
print(df) #must check and clean data before working on model

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price) # the x values are [[]] and are the features while the y is the target

print('Slopes:', reg.coef_)
print('Intercept:', reg.intercept_)
print('Prediction:', reg.predict([[3000, 3, 40]]))
print('Prediction 2:', reg.predict([[2500, 4, 5]]))