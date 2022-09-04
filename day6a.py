# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:47:23 2022

@author: vikas
"""

# Logistic Regression


X= list(range(-6,7))
X

y = []

import math

for x in X:
    y.append((1)/(1 + math.exp(-x)))
y


import matplotlib.pyplot as plt

plt.scatter(X,y)


import numpy as np
x= np.array(list(range(-12,13,2))).reshape((-1,1))
x.shape

y = np.array([0,0,0,0,1,1,0,0,1,1,1,1,1])
y.shape


plt.scatter(x,y)


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

pred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,pred)

x

ypred = model.predict(np.array([-1, 1]).reshape((-1,1)))

ypred


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x,y)

ypred = model.predict(x)

pred = model.predict_proba(x)

plt.xticks(range(-12,13))
plt.scatter(x,y)
plt.scatter(x,pred[:,1])
plt.scatter(x,ypred)

ypred1 = model.predict(np.array([-3,-2,-1, 1,2,5]).reshape((-1,1)))
ypred1



# Case binary.csv

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('25_LogR/binary.csv')

df.columns

x = df['gre'].values.reshape((-1,1))
y = df['admit'].values

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x,y)

ypred = model.predict(x)
ypredprob = model.predict_proba(x)

plt.scatter(x,y)
plt.scatter(x,ypred)
plt.scatter(x,ypredprob[:,1])



# Case Titanic

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('25_LogR/titanic.csv')

df.columns

x = df.drop(['survived'],axis=1).values
y = df['survived'].values
y

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x,y)

ypred = model.predict(x)
ypredprob = model.predict_proba(x)

ypred



y = [1,1,1,0,0,1,0,0,1,0] #Truth
p = [1,1,0,1,0,0,1,0,1,0] #Result

TP = 3

FN = 2

FP = 2

TN = 3

Accuracy = (TP+TN) / (TP+FN+FP+TN)

Accuracy

from sklearn.metrics import accuracy_score
accuracy_score(y,p)


y = [1,0,0,1,0,1,1,1,1,1] #Truth
p = [1,0,0,0,1,1,1,1,1,1] #Result

accuracy_score(y,p)

from sklearn.metrics import classification_report

print(classification_report(y,p))




import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('25_LogR/titanic.csv')

df.columns

x = df.drop(['survived'],axis=1).values
y = df['survived'].values
y

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x,y)

ypred = model.predict(x)
ypredprob = model.predict_proba(x)

ypred


from sklearn.metrics import classification_report

print(classification_report(y, ypred))


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest, ypred))





# HR Anlytics

import pandas as pd
df = pd.read_csv('25_LogR/HRNum.csv')
df.columns
x = df.drop(['Attrition'], axis=1).values
y = df['Attrition'].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest, ypred))



#Data Balancing

df['Attrition'].value_counts()

import numpy as np
from imblearn.over_sampling import SMOTE
sampler = SMOTE()
Xr, yr = sampler.fit_resample(x, y)

np.sum(yr==1)
np.sum(yr==0)


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain,ytest = train_test_split(Xr,yr,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest, ypred))









































































































