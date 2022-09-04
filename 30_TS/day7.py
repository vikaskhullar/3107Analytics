# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:34:01 2022

@author: vikas
"""


import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

#from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm


import matplotlib.pyplot as plt



dates = pd.date_range('2012-07-09','2012-07-30')

dates

series = [43.,32.,63., 98.,65.,78.,23., 35.,78.,56.,45., 45.,56.,6.,63.,45., 64.,34.,76.,34., 14., 54. ]

res = pd.Series(series, index=dates)
res

plt.plot(res)


#r = ARIMA(res, (3,0,3))
r = sm.tsa.arima.ARIMA(res, order=(3,0,3))

r.fit()

st = len(res)
en = len(res)+10

pred = r.predict(start=st, end=en, dynamic= True)



'''
pip install tsfresh


from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh.feature_extraction import ComprehensiveFCParameters, settings



# Reading the data

data.columns = ['month','#Passengers']
data['month'] = pd.to_datetime(data['month'],infer_datetime_format=True,format='%y%m')
df_pass, y_air = make_forecasting_frame(data["#Passengers"], kind="#Passengers", max_timeshift=12, rolling_direction=1)
print(df_pass)

'''
'''
# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(data, 'month', '#Passengers')
# Set aside the last 36 months as a validation series
train, val = series[:-36], series[-36:]
'''


#from darts.models import ExponentialSmoothing
#model = ExponentialSmoothing()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries

dates = pd.date_range('2012-07-09','2012-08-09')
series = [43.,32.,63., 98.,65.,78.,23., 35.,78.,56.,45., 45.,56.,6.,63.,45., 64.,34.,76.,34., 14., 54.,56.,6.,63.,45., 64.,34.,76.,34., 14., 54. ]

res = pd.DataFrame({'Dates':dates, 'Series':series})
res



#Loading the package
from darts import TimeSeries
#from darts.models import ExponentialSmoothing
from darts.models.forecasting.arima import ARIMA


series = TimeSeries.from_dataframe(res, 'Dates', 'Series')

#plt.plot(res)


model = ARIMA (p=10,d=0,q=5)


model.fit(series)


prediction = model.predict(len(series), num_samples=100)

prediction

series.plot()
prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
plt.legend()




#Case Air Passenge

df = pd.read_csv('AirPassengers.csv')
df

df['#Passengers'] = df['#Passengers'].fillna(method='ffill')



series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')

#plt.plot(res)

model = ARIMA (p=5,d=1,q=5)


model.fit(series)


prediction = model.predict(len(series), num_samples=100)

prediction

series.plot()
prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
plt.legend()




import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from statsmodels.tsa.seasonal import seasonal_decompose 



airline = pd.read_csv('AirPassengers.csv', 
					index_col ='Month', 
					parse_dates = True) 


airline.head() 

airline['#Passengers'] = airline['#Passengers'].fillna(method='ffill')

result = seasonal_decompose(airline['#Passengers'], model ='multiplicative') 

result.plot() 




from statsmodels.tsa.statespace.sarimax import SARIMAX 

# Train the model on the full dataset 
model = SARIMAX(airline['#Passengers'], 
						order = (3, 0, 3), 
						seasonal_order =(3, 0, 3, 12)) # 4 for Quartely, 12 for Monthly, 52 for Weakly
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(airline), 
						end = (len(airline)) + 3 * 12, 
						typ = 'levels').rename('Forecast') 




forecast

# Plot the forecast values 

forecast1 = result.predict(start = 0, 
						end = 144, 
						typ = 'levels').rename('Forecast1')


airline['#Passengers'].plot(figsize = (12, 5), legend = True, lw=4, linestyle='dashed') 
forecast.plot(legend = True, lw=4) 
forecast1.plot(legend = True,lw=2) 



# Stock Market Data

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime

pd.options.display.max_rows=50
pd.options.display.max_columns=10

enddate = datetime.datetime.today().date()
startdate = datetime.datetime.today().date() - datetime.timedelta(days=360)


df = data.DataReader('AWL.NS', 'yahoo', startdate, enddate)

df = df['Close']

df


dat = pd.date_range(start=startdate, end = enddate)
dat

df =df.reindex(dat)

df

df = df.fillna(method='ffill')

df

df.plot()



df 

cdf = pd.DataFrame(df)

cdf.columns = ['price']

cdf

from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(cdf['price'], model ='multiplicative') 

# ETS plot 
result.plot() 



import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(cdf, order=(3,1,3), seasonal_order=(3,1,3,12))
results = mod.fit()
results


from datetime import timedelta

pre = 60
pred  = results.get_prediction(start = len(cdf), end=len(cdf) + pre, dynamic=False)

#pred  = results.get_prediction(start = end_date, end=end_date + timedelta(days=pre), dynamic=False)

pred

pred_ci = pred.conf_int()
pred_ci

prediction=pred.predicted_mean
prediction


plt.plot(cdf)
plt.plot(pred_ci)
plt.plot(prediction)




# Rain Case Study

import pandas as pd
df = pd.read_csv('rain.csv', index_col ='Date', parse_dates=True)
df.dtypes
df

dat = pd.date_range(start='1990-01-06' , end = '2016-12-31')
dat

df =df.reindex(dat)

sum(df['RF'].isna())


df = df.fillna(0)

df

df.plot()
df[df['RF']<0]=0


df.columns


cdf = df.iloc[0:1000]

from statsmodels.tsa.seasonal import seasonal_decompose 
result = seasonal_decompose(cdf['RF']) 
result.plot() 


import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(cdf, order=(3,1,3), seasonal_order=(3,1,3,12))
results = mod.fit()
results


from datetime import timedelta

pre = 60
pred  = results.get_prediction(start = len(cdf), end=len(cdf) + pre, dynamic=False)

#pred  = results.get_prediction(start = end_date, end=end_date + timedelta(days=pre), dynamic=False)

pred

pred_ci = pred.conf_int()
pred_ci

prediction=pred.predicted_mean
prediction


import matplotlib.pyplot as plt
plt.plot(cdf)
plt.plot(pred_ci)
plt.plot(prediction)


from darts import TimeSeries
from darts.models.forecasting.arima import ARIMA


cdf['sno'] = cdf.index


series = TimeSeries.from_dataframe(cdf, 'sno', 'RF')

#plt.plot(res)


model = ARIMA (p=5,d=0,q=5)


model.fit(series)


prediction = model.predict(len(series), num_samples=100)

prediction

series.plot()
prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
plt.legend()




import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

n = 480
ts = pd.Series(np.random.randn(n), index=pd.date_range(start="2014-02-01", periods=n, freq="H"))


fig, ax = plt.subplots(figsize=(12,5))
seaborn.boxplot(df)



#Neural Prophet provided by facebook or Meta




import pandas as pd
df = pd.read_csv('rain.csv', index_col ='Date', parse_dates=True)
df.dtypes
df

dat = pd.date_range(start='1990-01-06' , end = '2016-12-31')
dat

df =df.reindex(dat)

sum(df['RF'].isna())


df = df.fillna(0)

df

df.plot()
df[df['RF']<0]=0


df.columns


cdf = df



'''
df = pd.read_csv('austin_weather.csv')
df.tail()

df.Date.unique()

df ['Date'] = pd.to_datetime(df ['Date'])
df.tail()
'''

cdf.columns

cdf['Date'] = cdf.index

import matplotlib.pyplot as plt
plt.plot(cdf ['Date'], cdf ['RF'])
plt.show()


new_column = cdf[['Date', 'RF']] 
new_column.dropna(inplace=True)
new_column.columns = ['ds', 'y'] 
new_column.tail()



import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt

n = NeuralProphet()
model = n.fit(new_column, freq='D')

cdf

future = n.make_future_dataframe(new_column, periods=3500)
forecast = n.predict(future)
forecast.tail()

plot = n.plot(forecast)

plt.plot(cdf['Date'][1:355], cdf['RF'][1:355])





##Neural Prophet provided by facebook or Meta


import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt

df = pd.read_csv('austin_weather.csv')
df.tail()

df.Date.unique()

df ['Date'] = pd.to_datetime(df ['Date'])
df.tail()


plt.plot(df ['Date'], df ['TempAvgF'])
plt.show()


new_column = df[['Date', 'TempAvgF']] 
new_column.dropna(inplace=True)
new_column.columns = ['ds', 'y'] 
new_column.tail()


n = NeuralProphet()
model = n.fit(new_column, freq='D')


future = n.make_future_dataframe(new_column, periods=1500)
forecast = n.predict(future)
forecast.tail()

plot = n.plot(forecast)



#

import numpy as np
cdf['RF'][cdf['RF']==0] = np.nan
cdf

cdf['RF'] = cdf['RF'].fillna(method='ffill')

cdf['RF'].plot()

sum(cdf['RF']==0)


cdf.columns
cdf['Date'] = cdf.index

import matplotlib.pyplot as plt
plt.plot(cdf ['Date'], cdf ['RF'])
plt.show()


new_column = cdf[['Date', 'RF']] 
new_column.dropna(inplace=True)
new_column.columns = ['ds', 'y'] 
new_column.tail()



import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt

n = NeuralProphet()
model = n.fit(new_column, freq='D')

cdf

future = n.make_future_dataframe(new_column, periods=3500)
forecast = n.predict(future)
forecast.tail()

plot = n.plot(forecast)

plt.plot(cdf['Date'][1:355], cdf['RF'][1:355])





LSTM
































































