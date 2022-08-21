# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:10:45 2022

@author: vikas
"""


import matplotlib.pyplot as plt

Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Unemployment_Rate1 = [1.8,2,8,4.2,6.0,7,3.5,5.2,7.5,5.3]

plt.plot(Year, Unemployment_Rate)
plt.title('Year vs Unemployment_Rate')
plt.xlabel('Year')
plt.ylabel('Unemployment_Rate')
plt.show()




plt.plot(Year, Unemployment_Rate, c = 'red', label='UR1', marker= 'o', markersize='10')
plt.plot(Year, Unemployment_Rate1,c = 'blue', label='UR2', marker = '>', markersize='10')
plt.title('Year vs Unemployment_Rate', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment_Rate', fontsize=14)
plt.legend()
plt.show()



cp = ['#A0E7E5', '#B4F8C8', '#FBE7C6', '#FFAEBC']

plt.plot(Year, Unemployment_Rate, c = cp[0], label='UR1', marker= 'o', markersize='10')
plt.plot(Year, Unemployment_Rate1,c = cp[2], label='UR2', marker = '>', markersize='10')
plt.title('Year vs Unemployment_Rate', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment_Rate', fontsize=14)
plt.yticks(range(0,16,2))
plt.legend()
plt.show()


import pandas as pd

Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], 'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]}
Data  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
df


plt.plot(df['Year'], df['Unemployment_Rate'], c = cp[0], label='UR1', marker= 'o', markersize='10')
plt.title('Year vs Unemployment_Rate', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment_Rate', fontsize=14)
plt.yticks(range(0,16,2))
plt.legend()
plt.show()



Country = ['USA','Canada','Germany','UK','France']
GDP_Per_Capita = [45000,42000,52000,49000,47000]


col = ['r', 'g', 'b', 'y']
plt.bar(Country, GDP_Per_Capita, color=col)
plt.title("Country vs GDP_Per_Capita")
plt.xlabel('Country')
plt.ylabel('GDP_Per_Capita')
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('tips')

df.head(5)

plt.scatter(df['total_bill'], df['tip'])
plt.title("Total Bill vs Tip")
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()



from pydataset import data

df = data('mtcars')

df.head(1)

plt.scatter(df['mpg'], df['hp'])


plot  = plt.scatter(df['mpg'], df['hp'], c = df['cyl'], label=df['cyl'])
plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(df['cyl'].values)))
plt.title("MTCARS")
plt.xlabel('MPG')
plt.ylabel('HP')
plt.savefig('scatter.jpeg')
plt.show()



#Multiple Graphs

df.head(1)
fig, ax = plt.subplots(1,2, dpi=300)
ax[0].scatter(df['mpg'], df['hp'])
ax[0].set_title('MPG vs HP')
ax[1].scatter(df['mpg'], df['disp'])
ax[1].set_title('MPG vs Disp')
plt.savefig('scatter2.jpeg')



# Distribution Ploting

import numpy as np
import matplotlib.pyplot as plt

d1 = np.random.normal(100, 10, size=10000)

plt.hist(d1)
plt.scatter(range(0, len(d1)), d1)

np.mean(d1)
np.median(d1)
np.std(d1)


plt.boxplot(d1)







import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('tips')
plt.scatter(df['total_bill'], df['tip'])

plt.boxplot(df['total_bill'])
plt.boxplot(df['tip'])



import seaborn as sns
sns.distplot(df['total_bill'], kde=True)




labels = ['BBA', 'MBA','PHD']
sizes=[120,30,10]
plt.pie(sizes, labels=labels, autopct='%1.3f%%', shadow=False)
plt.show()




import matplotlib.pyplot as plt
labels = ['Male',  'Female']
percentages = [60, 40]
explode=(0.15,0)
#

color_palette_list = ['#f600cc', '#ADD8E6', '#63D1F4', '#0EBFE9', '#C1F0F6', '#0099CC']

fig, ax = plt.subplots()
ax.pie(percentages, explode=explode, labels=labels, colors= color_palette_list, autopct='%1.2f%%',  shadow=True, startangle=90,  pctdistance=1.2, labeldistance=1.4)
ax.axis('equal')
ax.set_title("Distribution of Gender in Class", y=1)
ax.legend(frameon=False, bbox_to_anchor=(0.2,0.8))
plt.show()







































