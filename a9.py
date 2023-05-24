import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 #read dataset
df=pd.read_csv('titanic.csv')

 #display information of dataset
print('Titanic dataset is loaded successfully...')
print('Information of Dataset:\n',df.info)
print('Shape of Dataset:\n',df.shape)
print('Columnsname:n',df.columns)
print('Total elements in Dataset:',df.size)
print('Datatype of attributes(columns):',df.dtypes)
print('First 5 rows:',df.head())
print('Last 5 rows:',df.tail())

#find null values
print('Total number of  null values in Dataset:',df.isna().sum())

#fill null values
df['Age'].fillna(df['Age'].median(),inplace=True)
print(df.isna().sum())

#boxplot of 1-variable
fig,axes=plt.subplots(1,2)
fig.suptitle('boxplot of 1 variable (age&fare)')
sns.boxplot(data=df,x='Age',ax=axes[0])
sns.boxplot(data=df,x='Fare',ax=axes[1])
plt.show()

#boxplot of 2-variable
fig,axes=plt.subplots(2,2)
fig.suptitle('boxplot of 2 variables')
sns.boxplot(data=df,x='Survived',y='Age',hue='Survived',ax=axes[0,0])
sns.boxplot(data=df,x='Survived',y='Fare',hue='Survived',ax=axes[0,1])
sns.boxplot(data=df,x='Sex',y='Age',hue='Sex',ax=axes[1,0])
sns.boxplot(data=df,x='Sex',y='Fare',hue='Sex',ax=axes[1,1])

#boxplot of 3-variable
fig,axes=plt.subplots(1,2)
fig.suptitle("Boxplot of 3 variables")
sns.boxplot(data=df,x='Sex',y='Age',hue='Survived',ax=axes[0])
sns.boxplot(data=df,x='Sex',y='Age',hue='Survived',ax=axes[1])
plt.show()