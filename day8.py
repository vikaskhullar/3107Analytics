# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 19:36:33 2022

@author: vikas
"""

from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ec4604b2a608472dbf57f26b938ed8b9')

# /v2/top-headlines
th = newsapi.get_top_headlines(q='NSE',language='en')
th['totalResults']

type(th)

th.keys()

th['status']

type(th['articles'])

len(th['articles'])

th['articles'][0]

type(th['articles'][0])


th['articles'][0].keys()

text = th['articles'][0]['description']


des = []
score = []


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()



for art in th['articles']:
    text = art['description']
    des.append(text)
    score.append(sid.polarity_scores(text)['compound'])

score
des

import pandas as pd
df = pd.DataFrame({"Des":des,"Score":score})

df.to_csv("NewsSentiment.csv")


# /v2/top-headlines/sources
sources = newsapi.get_sources()




# Twitter Analysis


import tweepy

APIKey='CBoTcQQdaVbX4BHUIdo7JQTLW'
APISecret='xvHUgxzxehbnfPZHZPwo0xrJwxIgeGtt3YXXTs6qOOQmHOsOAB'
AccessToken='144501392-D78GmsUZPFtA4grDXUqYcVDb5btO67OqHOCoJ3R1'
AccessTokenSecret='gQixFplGbTkX6ueCLfMya71fs9rQZTxdhi8BX2IZL1mzC'


auth = tweepy.OAuthHandler(APIKey, APISecret)
auth.set_access_token(AccessToken, AccessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)



handle =['Adani']


desc = []
created = []

for tweets in api.search_tweets(q=handle, count = 100, lang="en"):
    desc.append(tweets.text)
    created.append(tweets.created_at)

desc


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


score=[]
for text in desc:
    score.append(sid.polarity_scores(text)['compound'])



df = pd.DataFrame({'CreatedAt':created, 'Description':desc, 'Score':score})
df.to_csv('TwiterRes.csv')



'''
tweets.created_at
tweets.text

    print(tweets.created_at, tweets.text.encode('utf-8'))



    creat.append(tweets.created_at)
    txt.append(tweets.text)

'''


# Time Series Analysis

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

#Auto Regressor, Integrated, Moving Average
'''
p: AR, derivative for lag observations
d: Number of times that the raw observations are differenced; also known as the degree of differencing.
q: MA window size
'''
'''
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('AirPassengers1.csv')
df=df.fillna(method='ffill')
df.plot()



df = df.dropna()

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

#p d q


len(df)

dt = pd.date_range('1949-01-01','1961-01-01', freq='m')

len(dt)

df.columns
data=df['#Passengers'].values.astype('int')

r = ARIMA(data, (3,0,0))

r = r.fit()

pred = r.predict(1, 144)

ypred = r.predict(1, 190)
len(pred)

df.columns
plt.plot(range(0,144), df['#Passengers'])
plt.plot(range(0,144), pred)
plt.plot(range(0,190), ypred)

'''



import pmdarima as pmd
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('AirPassengers1.csv')
df=df.fillna(method='ffill')

res = df['#Passengers']

import pmdarima as pmd
arima_model = pmd.auto_arima(res, 
                              start_p=1, 
                              start_q=1)

pred = arima_model.predict(40)
pred

plt.plot(res)
plt.plot(range(144, 144+len(pred)), pred)





from statsmodels.tsa.seasonal import seasonal_decompose 
airline = pd.read_csv('AirPassengers1.csv', 
					index_col ='Month', 
					parse_dates = True) 

airline = airline.fillna(method='ffill')
# Print the first five rows of the dataset 
airline.head() 

# ETS Decomposition 
result = seasonal_decompose(airline['#Passengers'], model ='multiplicative') 
result.plot() 




from statsmodels.tsa.statespace.sarimax import SARIMAX 

model = SARIMAX(airline['#Passengers'], 
						order = (2, 1, 1), 
						seasonal_order =(2, 1, 1, 12)) # 4 for Quartely, 12 for Monthly, 52 for Weakly
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(airline), 
						end = (len(airline)) + 10 * 12, 
						typ = 'levels').rename('Forecast') 

forecast

plt.plot(range(0,len(airline['#Passengers'])), airline['#Passengers'])
plt.plot(range(144,144+len(forecast)), forecast)


# Stock Market Analysis


from pandas_datareader import data

import datetime

start_date = datetime.datetime.today().date() - datetime.timedelta(days=91)

start_date

end_date = datetime.datetime.today().date()

panel_data = data.DataReader('AWL.NS', 'yahoo', start_date, end_date)
panel_data.columns


close = panel_data['Close']
close
close.plot()

all = pd.date_range(start=start_date, end=end_date)
all

close = close.reindex(all)

close[1:20]

close = close.fillna(method='ffill')
close = close.fillna(method='bfill')


close_df= pd.DataFrame(close)
close_df
close_df.columns =['price'] 

from statsmodels.tsa.seasonal import seasonal_decompose 
result = seasonal_decompose(close_df['price'].iloc[1:15], model ='multiplicative') 
# ETS plot 
result.plot() 


import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(close_df, order=(3,1,3), seasonal_order=(3,1,3,12))
res = mod.fit()
pred = res.predict(start=len(close_df), end = len(close_df)+20)
pred

plt.plot(range(1, len(close_df)+1), close_df['price'])
plt.plot(range(len(close_df), len(close_df)+len(pred)), pred)


import pmdarima as pmd
arima_model = pmd.auto_arima(close_df, 
                              start_p=1, 
                              start_q=1)

pred = arima_model.predict(40)
pred


plt.plot(close_df)
plt.plot(range(len(close_df), len(close_df)+len(pred)), pred)



# Crypto Market Analysis


from pandas_datareader import data

import datetime

start_date = datetime.datetime.today().date() - datetime.timedelta(days=91)

start_date

end_date = datetime.datetime.today().date()

panel_data = data.DataReader('BTC-USD', 'yahoo', start_date, end_date)
panel_data.columns


close = panel_data['Close']
close
close.plot()

all = pd.date_range(start=start_date, end=end_date)
all

close = close.reindex(all)

close[1:20]

close = close.fillna(method='ffill')
close = close.fillna(method='bfill')


close_df= pd.DataFrame(close)
close_df
close_df.columns =['price'] 

from statsmodels.tsa.seasonal import seasonal_decompose 
result = seasonal_decompose(close_df['price'], model ='multiplicative') 
# ETS plot 
result.plot() 


import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(close_df, order=(3,1,3), seasonal_order=(3,1,3,12))
res = mod.fit()
pred = res.predict(start=len(close_df), end = len(close_df)+30)
pred

plt.plot(range(1, len(close_df)+1), close_df['price'])
plt.plot(range(len(close_df), len(close_df)+len(pred)), pred)


import pmdarima as pmd
arima_model = pmd.auto_arima(close_df, 
                              start_p=1, 
                              start_q=1)

pred = arima_model.predict(40)
pred

plt.plot(close_df)
plt.plot(range(len(close_df), len(close_df)+len(pred)), pred)


#RFM Analysis


import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt

data = pd.read_csv('RFM/OnlineRetail.csv', parse_dates=['InvoiceDate'], encoding= 'unicode_escape')
data.dtypes

data.head(5)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data.head(5)

data= data[pd.notnull(data['CustomerID'])]

data['CustomerID'] = data['CustomerID'].astype('int').astype('object')

data.columns

data = data.drop(['StockCode', 'Description'], axis=1)

data.columns

data['Country'].value_counts()

ukdata = data[data['Country']=='United Kingdom']


ukdata['InvoiceDate'].dtype

ukdata['InvoiceDate'] = pd.to_datetime(ukdata['InvoiceDate'])
ukdata['InvoiceDate'].dtype
ukdata.columns
ukdata['TotalPrice'] = ukdata['Quantity'] * ukdata['UnitPrice'] 

#RFM

PRESENT = dt.datetime(2011,12,10)


ld = lambda date: (PRESENT - date).days

ser = ukdata['InvoiceDate'].apply(ld)

ser[1:10]


rfm = ukdata.groupby('CustomerID').agg(
    {'InvoiceDate': lambda date: (PRESENT - date.max()).days,
     'InvoiceNo': lambda num: len(num),
     'TotalPrice': lambda price: price.sum()})

rfm.head()

rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.head()


rfm['r_quartile'] = pd.qcut(rfm['Recency'], 3, ['1','2','3'])
rfm['f_quartile'] = pd.qcut(rfm['Frequency'], 3, ['3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['Monetary'], 3, ['3','2','1'])


rfm.head()


rfmResult = rfm[['r_quartile', 'f_quartile','m_quartile']]
rfmResult.head()
rfmResult.columns = ['R','F','M']
rfmResult.head()
rfmResult['RFM'] = rfmResult['R'].astype('string')+rfmResult['F'].astype('string')+rfmResult['M'].astype('string')
rfmResult = rfmResult.drop(['R','F','M'], axis=1)
rfmResult['RFM'].head(10)
rfmResult[rfmResult['RFM']=='331']
























































# Case Study

from pandas_datareader import data

import datetime

start_date = datetime.datetime.today().date() - datetime.timedelta(days=91)

start_date

end_date = datetime.datetime.today().date()

panel_data = data.DataReader('AWL.NS', 'yahoo', start_date, end_date)

panel_data.columns






