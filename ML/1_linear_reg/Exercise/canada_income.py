# canada_income.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

#read the file
df = pd.read_csv('canada_per_capita_income.csv')

# clean the data
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.strip('$')

#plot the graph

plt.title('Per Capita income per year')
plt.ylabel('Per-Capita Income')
plt.xlabel('Year')
plt.scatter(df.year, df.per_capita_income_us)

# fit and use linear_model

reg = linear_model.LinearRegression() # reg = Regression line
reg.fit(df[['year']], df.per_capita_income_us) # the x value must be a [[]] and the y value is just the y 

with open('reg_pickle', 'wb') as f: # wb = writing-binary
    pickle.dump(reg, f) # dump means to save file
    
with open('reg_pickle','rb') as f: #rb = reading-binary
    pickle_reg = pickle.load(f)
    
# I can either use pickle or use joblib to save. use joblib for large scale numpy arrays

import joblib

joblib.dump(reg, 'reg_joblib')
joblib_reg = joblib.load('reg_joblib')    
    


plt.plot(df.year, reg.predict(df[['year']]), color = 'yellow')

print('Slope:', reg.coef_)
print('Interept:', reg.intercept_)
print('Prediction for 2020:', reg.predict([[2020]]))