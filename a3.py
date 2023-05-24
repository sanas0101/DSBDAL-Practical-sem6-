
import pandas as pd
df= pd.read_csv('Employee_Salary_Dataset.csv')
print('\033[1m Employees Dataset is successfully loaded.......\033[0m\n')
df2=pd.read_csv('iris.csv')
print('Iris Dataset is successfully loaded....')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#first
print('1.Display information of Employees_Salary Dataset')

print('Information of Dataset:\n',df.info)
print('Shape of Dataset (row x column:)',df.shape)
print('Columns Name:',df.columns)
print('Total elements in dataset:',df.size)
print('Datatype of attribute (columns):',df.dtypes)
print('First 5 rows:\n',df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)

#second
print('2.Display Statistical information of Numerical Columns of Employees_SalaryDataset')

print('Statistical information of Numerical Columns: \n',)
columns=['Experience_Years','Age','Salary']
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns','Min','Max','Mean','Median','STD'))
for column in columns:
 m1,m2,m3=df[column].min(),df[column].max(),df[column].mean()
 m4,s=df[column].median(),df[column].std()
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format(column,m1,m2,m3,m4,s))


#third
print('3.Groupwise Stastistical of Employees_Salary Dataset')

print('Groupwise Statistical Summary....')
columns=['Experience_Years','Age','Salary']
for column in columns:
 print('\n-----------------------------',column,'---------------------------\n')
 print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns','Min','Max','Mean','Median','STD'))
m1=df[column].groupby(df['Gender']).min()
m2=df[column].groupby(df['Gender']).max()
m3=df[column].groupby(df['Gender']).mean()
m4=df[column].groupby(df['Gender']).median()
s=df[column].groupby(df['Gender']).std()
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Female',m1[0],m2[0],m3[0],m4[0],s[0]))
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Male',m1[1],m2[1],m3[1],m4[1], s[1]))

#fourth
print('4.Bar plot for all Statistics of Employees_Salary Dataset')

X=['min','max','mean','median','std']
features=['Salary','Age','Experience_Years']
df1 = pd.DataFrame(columns=['min','max','mean','median','std'])
for var in features:
 df1['min']=df[var].groupby(df['Gender']).min()
 df1['max']=df[var].groupby(df['Gender']).max()
 df1['mean']=df[var].groupby(df['Gender']).mean()
 df1['median']=df[var].groupby(df['Gender']).median()
 df1['std']=df[var].groupby(df['Gender']).std()
X_axis=np.arange(len(X))
plt.bar(X_axis-0.2,df1.iloc[0],0.4,label='Female')
plt.bar(X_axis+0.2,df1.iloc[1],0.4,label='Male')
plt.xticks(X_axis, X)
plt.xlabel('Statistical information')
plt.ylabel(var)
plt.title("Groupwise Statistical Information of Employees_Salary Dataset")
plt.legend()
plt.show()

#fifth
print('5.Display information of Iris Dataset')

df2= pd.read_csv('iris.csv')
print('Information of Dataset:\n',df2.info)
print('Shape of Dataset (row x column:)',df2.shape)
print('Columns Name:',df2.columns)
print('Total elements in dataset:',df2.size)
print('Datatype of attribute (columns):',df2.dtypes)
print('First 5 rows:\n',df2.head().T)
print('Last 5 rows:\n',df2.tail().T)
print('Any 5 rows:\n',df2.sample(5).T)

#sixth
print('6.Display Statistical information of Numerical Columns of Iris Dataset')

print('Statistical information of Numerical Columns: \n',)
columns=['sepal_length','sepal_width','petal_length','petal_width']
print("{:<25}{:<15}{:<15}{:<25}{:<15}{:<25}".format('Columns','Min','Max','Mean','Median','STD'))
for column in columns:
 m1,m2,m3=df2[column].min(),df2[column].max(),df2[column].mean()
 m4,s=df2[column].median(),df2[column].std()
print("{:<25}{:<15}{:<15}{:<25}{:<15}{:<25}".format(column,m1,m2,m3,m4,s))

#seventh
print('7.Groupwise Statistical of Iris Dataset')

print('Groupwise Statistical Summary....')
columns=['sepal_length','sepal_width','petal_length','petal_width']
for column in columns:
 print('\n-----------------------------',column,'---------------------------\n')
 print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns','Min','Max','Mean','Median','STD'))
m1=df2[column].groupby(df2['species']).min()
m2=df2[column].groupby(df2['species']).max()
m3=df2[column].groupby(df2['species']).mean()
m4=df2[column].groupby(df2['species']).median()
s=df2[column].groupby(df2['species']).std()
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Iris- setosa',m1[0],m2[0],m3[0],m4[0],s[0]))
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Iris- versicolor',m1[1],m2[1],m3[1],m4[1],s[1]))
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Iris- virginica',m1[0],m2[0],m3[0],m4[0],s[0]))

#eighth
print('8.Bar plot for all Statistics of Iris Dataset')

X=['min','max','mean','median','std']
features=['sepal_length','sepal_width','petal_length','petal_width']
df3 = pd.DataFrame(columns=['min','max','mean','median','std'])
for var in features:
 df3['min']=df2[var].groupby(df2['species']).min()
 df3['max']=df2[var].groupby(df2['species']).max()
 df3['mean']=df2[var].groupby(df2['species']).mean()
 df3['median']=df2[var].groupby(df2['species']).median()
 df3['std']=df2[var].groupby(df2['species']).std()
X_axis=np.arange(len(X))
plt.bar(X_axis-0.2,df3.iloc[0],0.3,label='Iris-setosa')
plt.bar(X_axis+0.1,df3.iloc[1],0.3,label='Iris-versicolor')
plt.bar(X_axis+0.4,df3.iloc[2],0.3,label='Iris-virginica')
plt.xticks(X_axis, X)
plt.xlabel('Statistical information')
plt.ylabel(var)
plt.title("Groupwise Statistical Information of Iris Dataset")
plt.legend()
plt.show()