# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 19:24:38 2022

@author: vikas
"""

import numpy as np

a = np.zeros((3,4))
a

b = np.ones((3,4))
b

n = np.random.randint(1, 10, size=(3,2)).astype('float')
n

c = np.zeros((3,4)).astype('int')
c

c = np.zeros((3,4)).astype('float')
c

d = np.eye(3,3)
d

n1 = np.linspace(0,10, num=5)
n1


n2 = np.random.randint(1, 10, size=(5,3,3))
n3 = np.zeros((3,3)).astype('int')
i=0
while(i<n2.shape[0]):
    n3 = n3 + n2[i]
    i=i+1
print(n3)



n2 = np.random.randint(1, 10, size=(5,3,3))
n3 = np.ones((3,3)).astype('int')
i=0
while(i<n2.shape[0]):
    n3 = n3 * n2[i]
    i=i+1
print(n3)



n4 = np.random.randint(1,50, size=(4,7))
n4

np.max(n4)
np.min(n4)
np.mean(n4)
np.median(n4)
np.std(n4)
np.var(n4)


#axis 1 for columns
np.max(n4, axis=0)
np.min(n4, axis=0)
np.mean(n4, axis=0)
np.median(n4, axis=0)
np.std(n4, axis=0)
np.var(n4, axis=0)


#axis 1 for rows
np.max(n4, axis=1)
np.min(n4, axis=1)
np.mean(n4, axis=1)
np.median(n4, axis=1)
np.std(n4, axis=1)
np.var(n4, axis=1)

n4


np.floor([1.2, 1.6])
np.ceil([1.2, 1.6])
np.trunc([1.2, 1.6])
np.round([1.2, 1.6])


np.floor([-1.2, -1.6])
np.ceil([-1.2, -1.6])
np.trunc([-1.2, -1.6])
np.round([-1.2, -1.6])


x = np.random.randint(1,10, size=5)
y = np.random.randint(11,20, size=5)

x
y

n5 = np.concatenate([x,y])
n5

n5 = np.concatenate([x,y], axis=0)
n5

x = np.random.randint(1,10, size=(3,5))
y = np.random.randint(11,20, size=(3,5))
x
y
n6 = np.concatenate([x,y], axis=0) # Joining Row after Row
n6
n7 = np.concatenate([x,y], axis=1) # Joining Row after Row
n7


n8 =  np.random.randint(1,20, size=(3,5))
n8
n8>10

np.sum(n8>10)

np.sum(n8>10, axis=0)
n8
n8.T

n9 =  np.random.randint(1,20, size=10)
n9
np.sort(n9)

x = np.random.randint(1,10, size=(3,3))
y = np.random.randint(11,20, size=(3,3))
x
y

np.multiply(x,y)
np.matmul(x,y)


x1 = x*2
x
x1



x = np.random.randint(1,10, size=(100,100))
y = np.random.randint(11,20, size=(100,100))
np.matmul(x,y)


#Pandas

import numpy as np
x = np.random.randint(1,10, size=(3,3))

x
i=0
while(i<x.shape[0]):
    x[i] = x[i]*(i+2)
    i=i+1
x


# Pandas
from pydataset import data
data('')
mt = data('mtcars')
type(mt)
mt


import pandas as pd
mt.to_csv('mt.csv')


import pandas as pd
df = pd.read_csv('mt.csv')
df
type(df)


df = pd.read_csv('train.csv')
df

df = pd.read_csv('winequality-white.csv', sep=';')
df


# pandas Series

r1 = range(10,16)
s = pd.Series(r1)
s

s[1]

s[3]


ps1 = pd.Series([1,3,5,7,8])
ps1

ps2 = pd.Series([1,3,5,7,8], index=[10,20,30,40,50])
ps2

ps3 = pd.Series([1,3,5,7,8], index=['a','b','c','d','e'])
ps3

ps4 = pd.Series([1,3,5,7,8], index=['a','a','c','d','d'])
ps4
ps4['a']
ps4.loc['a']

ps4.iloc[0]
ps4.iloc[1]
ps4.iloc[2]
ps4.iloc[3]
ps4.iloc[4]

ps4

ps4.index
ps4.values


ps4['c':'d']
ps4.iloc[2:4]
ps4.iloc[2:]


import numpy as np
n1 = np.random.randint(1,100, size=10)
pd5 = pd.Series(n1)
pd5

pd5>50
pd5[pd5>50]

(pd5>50) & (pd5<80)
pd5[(pd5>50) & (pd5<80)]
pd5


import pandas as pd
course = pd.Series(['BTech', 'MTech', 'MBA', 'BBA'])
strength = pd.Series([40,20,50,30])
fees = pd.Series([2.5, 1.5, 3.5, 2.7])
course
strength
fees
d1 ={'Course':course, 'Strength':strength, 'Fees':fees}
d1
df = pd.DataFrame(d1)

df
df.columns

df.loc[0:1]
df.iloc[0:1]

df.count()

df['Course']
df.Course

df.columns

df[['Course', 'Strength']]

df['Course']=='BTech'

type(df['Course'])

df[df['Course']=='BTech']

df

df[df['Fees']>2.5]
df[df['Fees']>2.5]['Course']

df.to_csv('student.csv')


# Missing Values


import pandas as pd
import numpy as np
placed = pd.Series([None,np.nan, 100, None])
placed

df['Placed'] = placed
df

#Row wise
df.dropna()
#axis 0 is Row wise
df.dropna(axis=0)

#axis 1 is Column wise
df.dropna(axis=1)



import pandas as pd
pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None], ['Vikas', None, None, None, None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',None,None],['upen',None,'M',None, None]])
pd4

#Droping
pd4.dropna(axis=0)
pd4.dropna(axis=1)
pd4

pd4.dropna(axis='rows')
pd4.dropna(axis='columns')

pd4

pd4.dropna(axis='rows', how = 'all')
pd4.dropna(axis='columns', how = 'all')


pd4.dropna(axis='rows', how = 'any')
pd4.dropna(axis='columns', how = 'any')

pd4

pd4.dropna(axis='rows', thresh = 3)
pd4.dropna(axis='columns', thresh=3)


#Filling

df.fillna(0)
pd4.fillna(0)



import pandas as pd
df = pd.read_csv('AirPassengers.csv')
df = df[:20]
df.plot()


df.fillna(0).plot()

df.fillna(method='ffill').plot()
df

df.fillna(method='bfill').plot()
df


'''
Data wrangling is the process of removing errors 
and combining complex data sets to make them more 
accessible and easier to analyze. 
'''

import pandas as pd
rno = pd.Series(range(1,11))
name = []
for i in range(1,11):
    name.append('Student'+str(i))
name

name = ['Student'+str(i) for i in range(1,11)]
name

gender = np.random.choice(['Male', 'Female'], size=10)


PMarks = np.random.randint(0, 100, size=10)
PMarks

MMarks = np.random.randint(0, 100, size=10)
MMarks

course = np.random.choice(['BTech', 'MTech', 'MSC', 'BSC'], size=10)

city = np.random.choice(['Delhi', 'Mumbai', 'Kolkata', 'Chennai'], size=10)


df = pd.DataFrame({'RNo':rno, 'Name':name, 'Gender':gender, 'PMarks':PMarks,'MMarks':MMarks, 'Course':course, 'City':city})

df




#I Million Dataset

import pandas as pd
rno = pd.Series(range(1,100001))
name = ['Student'+str(i) for i in range(1,100001)]
gender = np.random.choice(['Male', 'Female'], size=100000)
PMarks = np.random.randint(0, 100, size=100000)
MMarks = np.random.randint(0, 100, size=100000)
course = np.random.choice(['BTech', 'MTech', 'MSC', 'BSC'], size=100000)
city = np.random.choice(['Delhi', 'Mumbai', 'Kolkata', 'Chennai'], size=100000)
df = pd.DataFrame({'RNo':rno, 'Name':name, 'Gender':gender, 'PMarks':PMarks,'MMarks':MMarks, 'Course':course, 'City':city})
df

df.to_csv('StudentData.csv')




import pandas as pd
rno = pd.Series(range(1,11))
name = ['Student'+str(i) for i in range(1,11)]
gender = np.random.choice(['Male', 'Female'], size=10)
course = np.random.choice(['BTech', 'MTech', 'MSC', 'BSC'], size=10)
city = np.random.choice(['Delhi', 'Mumbai', 'Kolkata', 'Chennai'], size=10)
pd1 = pd.DataFrame({'RNo':rno, 'Name':name, 'Gender':gender, 'Course':course, 'City':city})
pd1




rno1 = pd.Series(range(6,21))
PMarks = np.random.randint(0, 100, size=15)
MMarks = np.random.randint(0, 100, size=15)

exam = pd.DataFrame({'RNo':rno1, 'PMarks':PMarks, 'MMarks':MMarks})
exam


set(pd1['RNo'])
set(exam['RNo'])
set(pd1['RNo']).intersection(set(exam['RNo']))

result = pd.merge(pd1, exam, how='inner')
result

set(pd1['RNo'])
set(exam['RNo'])
set(pd1['RNo']).union(set(exam['RNo']))

result = pd.merge(pd1, exam, how='outer')
result

result = pd.merge(pd1, exam, how='left')
result

result = pd.merge(pd1, exam, how='right')
result

exam1 = pd.DataFrame({'RNo1':rno1, 'PMarks':PMarks, 'MMarks':MMarks})
exam1

result = pd.merge(pd1, exam1, how='inner', left_on='RNo', right_on='RNo1')
result
result.drop('RNo1', axis=1)

name2 = ['Student'+str(i) for i in range(6,21)]
exam2 = pd.DataFrame({'RNo':rno1,'Name':name2, 'PMarks':PMarks, 'MMarks':MMarks})
exam2

result = pd.merge(pd1, exam2, how='inner')
result



'''
Data Fetching
Data Cleaning
Data Merging or Data Grouping if Required
Visualize Data
'''


import pandas as pd
df = pd.read_csv('calendar.csv')
df.columns
df.count()
df['price'].dropna()
df.head()
df.tail()
df.dropna()
df.fillna(0)
















































