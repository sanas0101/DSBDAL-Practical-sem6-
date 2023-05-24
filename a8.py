import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read dataset
df=pd.read_csv('titanic.csv')
print('Titanic dataset is loaded successfully...')
#

#information of dataset

print('Information of Dataset:\n',df.info)
print('Shape of Dataset:\n',df.shape)
print('Columnsname:n',df.columns)
print('Total elements in Dataset:',df.size)
print('Datatype of attributes(columns):',df.dtypes)
print('First 5 rows:',df.head())
print('Last 5 rows:',df.tail())

#display null values
print('Total number of values in Dataset:',df.isna().sum())

#fill null values
df['Age'].fillna(df['Age'].median(),inplace=True)
print(df.isnull().sum())


#histogram of 1 variable
fig,axes=plt.subplots(1,2)
fig.suptitle('Histogram of 1-variable')
sns.histplot(data=df,x='Age',ax=axes[0])
sns.histplot(data=df,x='Fare',ax=axes[1])
plt.show()

#histogram of 2 variable
fig,axes=plt.subplots(2,2)
fig.suptitle('Histogram of 2 variables')
sns.histplot(data=df,x='Age',hue='Survived',multiple='dodge',ax=axes[0,0])
sns.histplot(data=df,x='Fare',hue='Survived',multiple='dodge',ax=axes[0,1])
sns.histplot(data=df,x='Age',hue='Sex',multiple='dodge',ax=axes[1,0])
sns.histplot(data=df,x='Fare',hue='Sex',multiple='dodge',ax=axes[1,1])
plt.show()