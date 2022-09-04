# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 19:41:48 2022

@author: vikas
"""

# Decision Tree
# Random Forest
# K Nearest Neighbours



import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('25_LogR/titanic.csv')
df.columns
x = df.drop(['survived'],axis=1).values
y = df['survived'].values
y

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))





# Regression Tree


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

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.1)
x.shape, y.shape
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(ytest, pred)
r2


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(ytest, pred)
r2


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(ytest, pred)
r2



# Clustering


import matplotlib.pyplot as plt
import pandas as pd
data = {'x1': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'x2': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
df = pd.DataFrame(data,columns=['x1','x2'])
print (df.head())
plt.scatter(df['x1'], df['x2'])

from sklearn.cluster import KMeans
kmean = KMeans(n_clusters = 2)
kmean.fit(df)
centroids = kmean.cluster_centers_
centroids
labels = kmean.labels_.astype(float)
df['labels'] = labels
df
plt.scatter(df['x1'], df['x2'], c= labels, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")




kmean = KMeans(n_clusters = 30)
kmean.fit(df)
centroids = kmean.cluster_centers_
centroids
labels = kmean.labels_.astype(float)
df['labels'] = labels
df
plt.scatter(df['x1'], df['x2'], c= labels, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")


kmean.inertia_
sse=[]
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();




from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow






kmean = KMeans(n_clusters = 3)
kmean.fit(df)
centroids = kmean.cluster_centers_
centroids
labels = kmean.labels_.astype(float)
df['labels'] = labels
df
plt.scatter(df['x1'], df['x2'], c= labels, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")






import pandas as pd
df = pd.read_csv('Sensorless_drive_diagnosis.csv', header=None)
df.head(1)
dff = df.drop([48], axis=1)

kmean.inertia_
sse=[]

for k in range(1, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dff)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,20), sse)
plt.xticks(range(1,20))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();


from kneed import KneeLocator
kl = KneeLocator(x=range(1,20), y=sse, curve='convex', direction='decreasing')
kl.elbow


x = df.drop([48],axis=1).values
y = df[48].values
y



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))


#Case Food Cluster

df = pd.read_csv('28_Clustering/FoodCluster.csv')
df

kmean.inertia_
sse=[]

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,10), sse)
plt.xticks(range(1,10))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();


from kneed import KneeLocator
kl = KneeLocator(x=range(1,10), y=sse, curve='convex', direction='decreasing')
kl.elbow


kmean = KMeans(n_clusters = 3)
kmean.fit(df)
centroids = kmean.cluster_centers_
centroids
labels = kmean.labels_.astype(float)
df['labels'] = labels
df

df.to_csv('ResultFood.csv')




#Association Rule Mining

from efficient_apriori import apriori

transactions = [('eggs', 'bacon', 'soup', 'milk'), 
                ('eggs', 'bacon', 'apple', 'milk'), 
                ('soup', 'bacon', 'banana')]

transactions

itemsets, rules = apriori(transactions)

print(itemsets)
9
print(rules)


for rule in rules:
    print (rule)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


transactions = [['milk', 'water'], ['milk', 'bread'], ['milk','bread','water']]
transactions



te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df


frequent_itemsets = apriori(df, min_support=0.0000001, use_colnames = True)
frequent_itemsets

pd.set_option('display.max_columns',None)

res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000001)

print(res) #dataframe with confidence, lift, conviction and leverage metrics calculated


df_out = res[['antecedents', 'consequents', 'confidence', 'support', 'lift']]


df_out[df_out['confidence']==1]



# Case Study Store Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

store_data = pd.read_csv('29_Apriori\store_data1.csv', header=None)
store_data.head()

store_data.values.shape


records = []

for i in range(0, 7501):
    print(i)    
    records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])

records

from efficient_apriori import apriori
itemsets, rules = apriori(records, min_support=0.01, min_confidence=0.01)
len(rules)

for r in rules:
    print (r)



#MLXTEND


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

records


te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
te_ary.shape



df = pd.DataFrame(te_ary, columns=te.columns_)

df.to_csv('convert.csv')


frequent_itemsets = apriori(df, min_support=0.001, use_colnames = True)

confidence = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
print(confidence) #dataframe with confidence, lift, conviction and leverage metrics calculated
df1 = confidence[['antecedents', 'consequents', 'support','confidence']]

df1.to_csv('association.csv')



#Case 2

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('29_Apriori/online_store.csv')

df.head(1)

df.columns

df = df[['InvoiceNo', 'Description','Quantity']]

df['Description'] = df['Description'].str.strip()

df = df[~df['InvoiceNo'].str.contains('C')]
df = df[~df['InvoiceNo'].str.contains('A')]
df.dtypes
df

basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum()

basket



#Text analysis

# Modes for File Opening 1. Read, 2. Write, 3. Append

f = open('abc.txt', 'r')
#FileNotFoundError: [Errno 2] No such file or directory: 'abc.txt'


f = open('abc.txt', 'w')
f.write('Analytics Nearly Completed\n')
f.write('Sentiment analysis\n')
f.write('time Series Analysis\n')
f.close()


f = open('abc.txt', 'w')
f.close()




f = open('abc.txt', 'w')
f.write('Analytics Nearly Completed\n')
f.write('Sentiment analysis\n')
f.close()


f = open('abc.txt', 'a')
f.write('time Series Analysis\n')
f.close()



f = open('abc.txt', 'r')
text = f.read()
f.close()
print(text)


# CSV Writer

def csvwriter(dat, file):
    cnt=0
    f = open(file, 'a')
    for c in dat:
        f.write(c)
        if(cnt!=len(dat)-1):
            f.write(',')
        cnt=cnt+1
    f.write('\n')


csvwriter(['name','rno', 'class'], 'data.csv')
csvwriter(['AA','11', 'BB'], 'data.csv')
csvwriter(['BB','22', 'BBB'], 'data.csv')
csvwriter(['CC','33', 'CCC'], 'data.csv')




# Text Cleaning

filename = 'metamorphosis.txt'

file = open(filename, 'r')

text = file.read()
text
file.close()

text = text.replace('\n',' ')
text


import string

punc = string.punctuation
punc


for i in punc:
    print(i)
    text = text.replace(i,'')


text


# Sentiment Analysis

import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')



from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()

test1 = "I am good"
test2 = "This is bad"
test3 = "Bad worst Bad worst  Bad worst  Bad worst  Bad worst  Bad worst  Do not value students at all, provide the same course at different prices. Make false promises, which are not followed up.Would not recommend."

scores = sid.polarity_scores(test1)
scores

scores = sid.polarity_scores(test2)
scores

scores = sid.polarity_scores(test3)
scores
test4 = "I am happy with the way of teaching the Digital Marketing course by Henry Harvin. The course structure and internship options gave me a lot of knowledge to work as a digital marketer."

scores = sid.polarity_scores(test4)
scores


test5 = "I had a very worse experience in Henry Harvin. While admission they claim to be No.1 training institute, but after you have enrolled in any course, they don't give any importance to the customers. They give all false promises of giving internship after the course, but when we do enquire they don't response at all. They have the worst support team who give excuses every time. Teachers are good but for their pathetic support team, I don't recommend anyone to join this institute."

scores = sid.polarity_scores(test5)
scores










































































































