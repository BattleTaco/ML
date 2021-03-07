# true_hr.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load file
df = pd.read_csv('HR_comma_sep.csv')

# grab the data of people who quit and who stayed
left = df[df.left == 1]
stayed = df[df.left == 0]

#display the impact of salary and people who left or stayed
pd.crosstab(df.salary, df.promotion_last_5years).plot(kind = 'bar')
pd.crosstab(df.Department, df.left).plot(kind = 'bar')

