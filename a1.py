import pandas as pd
import numpy as np
import seaborn as sns
#read dataset
df=pd.read_csv('placement_data.csv')
print("Dataset is loaded successfully!!")

#display information of dataset
print('Information of Dataset:\n',df.info)
print('Shape of Dataset:\n',df.shape)
print('Columnsname:n',df.columns)
print('Total elements in Dataset:',df.size)
print('Datatype of attributes(columns):',df.dtypes)
print('First 5 rows:',df.head())
print('Last 5 rows:',df.tail())


#display statatistical information of dataset
print('Statistical information of Null values in Dataset:',df.describe())

#display null values
print('Total number of values in Dataset:',df.isna().sum())

#data type conversion
print('Converting Data type of variables:')
df['sl_no']=df['sl_no'].astype('int8')

print('Datatype of attributes(columns):',df.dtypes)


print('Gender values:',df['gender'].unique())
#5.1
print('b.Label Encoding Using cat.codes')

df['gender']=df['gender'].astype('category')
print('Data types of Gender=', df.dtypes['gender'])
df['gender']=df['gender'].cat.codes
print('Data types of gender after label encoding=',df.dtypes['gender'])
print('Gender Values :', df['gender'].unique())

#5.2
print('c.One Hot Encoding')

df=pd.get_dummies(df,columns=['gender'],prefix='sex')
print(df.head().T)

#5.3
print('d.Using Scikit-Learn Library')

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[['gender']]=enc.fit_transform(df[['gender']])
print(df.head().T)

#sixth
print('6.Normalisation using min-max scaling')

#6.1
print('a.Maximum Absolute Scaling')

df['salary']=df['salary']/df['salary'].abs().max()
print(df.head().T)

#6.2
print('b.Min-Max Feature Scaling')

df['salary']=(df['salary']-df['salary'].min()/(df['salary'].max()-
df['salary'].min()))
print(df.head().T)

#6.3
print('c.Z-Score method')

df['salary'] = (df['salary'] - df['salary'].mean()) 
(df['salary'].std())
print('\n z score is \n\n')
print(df['salary'].head().T)

#6.4
print('d.Robust Scaling')

df['salary'] = (df['salary'] - df['salary'].mean()) 
(df['salary'].quantile(0.75) - df['salary'].quantile())
print(df['salary'].head().T)

#6.5
print('e.Using Sci-kit learn')

from sklearn.preprocessing import MaxAbsScaler
abs_scaler=MaxAbsScaler()
df[['salary']]=abs_scaler.fit_transform(df[['salary']])
print('\n Maximum absolute Scaling method normalization -1 to 1 \n\n')
print(df['salary'].head().T)



