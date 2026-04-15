
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df1 = pd.read_csv('/canada_per_capita_income.csv')

# Feature and Target
X = df1[['year']]
y = df1['per capita income (US$)']

# Create model
model = LinearRegression()
model.fit(X, y)

# Predict for 2020
prediction_2020 = model.predict([[2020]])

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.title("Canada Per Capita Income Prediction")
plt.show()

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted per capita income in 2020:", prediction_2020[0])

df=pd.read_csv('/salary.csv')
df.isnull().sum()
df['YearsExperience'].fillna(df['YearsExperience'].mean(),inplace=True)
X=df[['YearsExperience']]
y=df['Salary']
model=LinearRegression()
model.fit(X,y)
ans=model.predict([[12]])
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Year")
plt.ylabel("Salary")
plt.title("Salary v/s Experience")
plt.show()

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 12 years:", ans[0])

import math
import pandas as pd

mean1 = df['YearsExperience'].mean()
mean2 = df['Salary'].mean()

num = 0
den = 0

for i in range(df.shape[0]):
    x = df.loc[i, 'YearsExperience']
    y = df.loc[i, 'Salary']

    num += (x - mean1) * (y - mean2)
    den += (x - mean1) ** 2

slope = num / den
intercept = mean2 - slope * mean1   # use *

print("Slope:", slope)
print("Intercept:", intercept)
print("Prediction for 12 years:", intercept + slope * 12)

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df2 = pd.read_csv('/hiring.csv')

# Convert word numbers to digits
word_to_num = {
    'zero':0, 'one':1, 'two':2, 'three':3, 'four':4,
    'five':5, 'six':6, 'seven':7, 'eight':8,
    'nine':9, 'ten':10, 'eleven':11, 'twelve':12
}

df2['experience'] = df2['experience'].map(word_to_num)

# Fill missing values
df2['experience'] = df2['experience'].fillna(0)
df2['test_score(out of 10)'] = df2['test_score(out of 10)'].fillna(
    df2['test_score(out of 10)'].mean()
)

# Features and target
X = df2[['experience','test_score(out of 10)','interview_score(out of 10)']]
y = df2['salary($)']

# Train model
model = LinearRegression()
model.fit(X,y)

# Predictions (use numbers now)
ans1 = model.predict([[2,9,6]])      # two years
ans2 = model.predict([[12,10,10]])   # twelve years

print("Salary for (2,9,6):", ans1)
print("Salary for (12,10,10):", ans2)

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df3 = pd.read_csv('/1000_Companies.csv')
df3.isnull().sum()
df3=pd.get_dummies(df3,columns=['State'],drop_first=True)
X=df3.drop('Profit',axis=1)
y=df3['Profit']
model=LinearRegression()
model.fit(X,y)
ans1=model.predict([[91694.48, 515841.3, 11931.24,1,0]])
print(ans1)
