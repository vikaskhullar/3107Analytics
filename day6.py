# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:29:17 2022

@author: vikas
"""

# Simple Linear Regression

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.array([5, 15, 25, 35, 45, 55])  # Feature
x.shape
x = x.reshape((-1, 1))
x.shape
y = np.array([5, 20, 14, 32, 22, 38])  # Label

y.shape


plt.scatter(x, y)


model = LinearRegression()

model.fit(x, y)

model.coef_
model.intercept_

pred = model.predict(x)

pred


plt.scatter(x, y)
plt.scatter(x, pred)
plt.plot(x, pred)

r2 = model.score(x, y)
r2


mse = mean_squared_error(y, pred)
mse


xpred = np.array([10, 50, 60])
xpred.shape
xpred

xpred = xpred.reshape((-1, 1))

xpred
xpred.shape

ypred = model.predict(xpred)

ypred


plt.scatter(x, y)
plt.scatter(x, pred)
plt.plot(x, pred)
plt.scatter(xpred, ypred)


# Case Housing


df = pd.read_csv('24_LR/house.csv')
df.columns
df
df =df.dropna()

x
x = df['area'].values.reshape((-1,1))
x.shape

y = df['Price'].values
y.shape

plt.scatter(x,y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

r2 = model.score(x,y)
r2

from sklearn.metrics import mean_squared_error

pred = model.predict(x)
pred

mse = mean_squared_error(y, pred)
mse

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y,pred)

mae

plt.scatter(x,y)
plt.scatter(x,pred)
plt.plot(x,pred)


xpred = np.array([2200]).reshape((-1,1))

ypred = model.predict(xpred)






# Case Housing1


df = pd.read_csv('24_LR/Housing.csv')
df.columns
df
df =df.dropna()

x
x = df['area'].values.reshape((-1,1))
x.shape

y = df['price'].values
y.shape

plt.scatter(x,y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

r2 = model.score(x,y)
r2

from sklearn.metrics import mean_squared_error

pred = model.predict(x)
pred

mse = mean_squared_error(y, pred)
mse

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y,pred)

mae

plt.scatter(x,y)
plt.scatter(x,pred)
plt.plot(x,pred)


xpred = np.array([7500, 8500]).reshape((-1,1))

ypred = model.predict(xpred)

ypred


#Outliers

plt.boxplot(df['area'])

df.columns

df = df[df['area']<10000]

df

plt.boxplot(df['area'])


plt.boxplot(df['price'])
df['price']


x = df['area'].values.reshape((-1,1))
x.shape

y = df['price'].values
y.shape

plt.scatter(x,y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

r2 = model.score(x,y)
r2

from sklearn.metrics import mean_squared_error
pred = model.predict(x)
pred

mse = mean_squared_error(y, pred)
mse

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y,pred)

mae

plt.scatter(x,y)
plt.scatter(x,pred)
plt.plot(x,pred)


xpred = np.array([7500, 8500]).reshape((-1,1))

ypred = model.predict(xpred)



#Stock data


df = pd.read_csv('24_LR/prices.csv')

df.columns
df = df.dropna()

x = df.drop(['volume'], axis=1).values
x.shape

y = df['volume'].values
y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

r2 = model.score(x,y)
r2
# Case dropped due to very low r2 score



# Case Life Expectancy Data
df = pd.read_csv('24_LR/Life Expectancy Data.csv')
df.columns
df = df.dropna()

x = df.drop(['Country', 'Year', 'Status', 'Life expectancy '], axis=1).values
x.shape
y = df['Life expectancy '].values
y.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

r2 = model.score(x,y)
r2


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3)

x.shape, y.shape
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(xtrain, ytrain)

pred = model.predict(xtest)


from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(ytest, pred)
r2

mse = mean_squared_error(ytest, pred)
mse

mae = mean_absolute_error(ytest, pred)
mae



# MT cars

from pydataset import data

df =  data('mtcars')
df.columns
df.head(2)

df = df[['mpg', 'disp', 'hp']]

df.columns

from statsmodels.formula.api import ols

model = ols("mpg ~ disp + hp", data = df).fit()

print(model.summary())

import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(model, fig=fig)




from pydataset import data
df =  data('mtcars')
df.columns
df.head(2)

df.columns

from statsmodels.formula.api import ols

model = ols("mpg ~ disp + cyl + drat + qsec + vs + am + gear + carb", data = df).fit()

print(model.summary())

import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8), dpi=300)
fig = sm.graphics.plot_ccpr_grid(model, fig=fig)
fig



# Case Life Expectancy Data
df = pd.read_csv('24_LR/Life Expectancy Data.csv')
df.columns
df = df.dropna()
df = df.drop(['Country', 'Year', 'Status'], axis=1)
df.columns




























