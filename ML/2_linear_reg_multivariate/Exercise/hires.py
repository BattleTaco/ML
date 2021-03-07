import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv('hiring.csv')

#clean data

df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)

median_test_score = math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
        
print(df)

# create model
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

print('Slopes:', reg.coef_)
print('Intercept:', reg.intercept_)

print('Prediction 1:', reg.predict([[2, 9, 6]]))
print('Prediction 2:', reg.predict([[12, 10, 10]]))