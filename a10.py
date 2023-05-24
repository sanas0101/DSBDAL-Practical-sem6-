import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
df=pd.read_csv('iris.csv')
#information of dataset
print(df.head)
print('Information of Dataset:\n',df.info)
print('Shape of Dataset:\n',df.shape)
print('Columnsname:n',df.columns)
print('Total elements in Dataset:',df.size)
print('Datatype of attributes(columns):',df.dtypes)
print('First 5 rows:',df.head())
print('Last 5 rows:',df.tail())

#display null values
print('Total number of values in Dataset:',df.isna().sum())

#histogram of 1-variable
fig,axes=plt.subplots(2,2)
fig.suptitle('Histogram 1-variables')
sns.histplot(data=df,x='sepal_length',ax=axes[0,0])
sns.histplot(data=df,x='sepal_width',ax=axes[0,1])
sns.histplot(data=df,x='petal_length',ax=axes[1,0])
sns.histplot(data=df,x='petal_width',ax=axes[1,1])
plt.show()
print(df.columns)

#histogram of 2-variable

fig,axes=plt.subplots(2,2)
fig.suptitle('Histogram of 2 variables')
df['species'] = df['species'].astype('category')

sns.histplot(data=df,x='sepal_length',hue='species',multiple='dodge',ax=axes[0,0])
sns.histplot(data=df,x='sepal_width',hue='species',multiple='dodge',ax=axes[0,1])
sns.histplot(data=df,x='petal_length',hue='species',multiple='dodge',ax=axes[1,0])
sns.histplot(data=df,x='petal_width',hue='species',multiple='dodge',ax=axes[1,1])
plt.show()

#boxplot of 1-varibale
fig,axes=plt.subplots(2,2)
fig.suptitle('Box-plot of 1- variables')
sns.boxplot(data=df,x='sepal_length',ax=axes[0,0])
sns.boxplot(data=df,x='sepal_width',ax=axes[0,1])
sns.boxplot(data=df,x='petal_length',ax=axes[1,0])
sns.boxplot(data=df,x='petal_width',ax=axes[1,1])
plt.show()


#histogram of 2-variable

fig,axes=plt.subplots(2,2)
fig.suptitle('Histogram of 2 variables')
df['species'] = df['species'].astype('category')

sns.boxplot(data=df,x='sepal_length',hue='species',ax=axes[0,0])
sns.boxplot(data=df,x='sepal_width',hue='species',ax=axes[0,1])
sns.boxplot(data=df,x='petal_length',hue='species',ax=axes[1,0])
sns.boxplot(data=df,x='petal_width',hue='species',ax=axes[1,1])
plt.show()
