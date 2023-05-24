def RemoveOutlier(df,var):
	Q1=df[var].quantile(0.25)
	Q3=df[var].quantile(0.75)
	IQR=Q3-Q1
	high=Q3+1.5*IQR
	low=Q1-1.5*IQR
	df=df[(df[var]>=low) & (df[var]<=high)]
	return df
def DisplayOutliers(df,message):
	fig,axes=plt.subplots(1,2)
	fig=plt.suptitle(message)
	sns.boxplot(data=df,x='Age',ax=axes[0])
	
	sns.boxplot(data=df,x='EstimatedSalary',ax=axes[1])
	plt.show()
#import libraries	

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#read dataset
df=pd.read_csv('Social_Network_Ads.csv')

#Display  infromation of dataset
print('Information of Dataset:\n',df.info)
print('Shape of Dataset:\n',df.shape)
print('Columnsname:n',df.columns)
print('Total elements in Dataset:',df.size)
print('Datatype of attributes(columns):',df.dtypes)
print('First 5 rows:',df.head())
print('Last 5 rows:',df.tail())

#find missing values
print('Missing Values:',df.isnull().sum())

#find correlation matrix
print('Finding correlation matrix using heatmap:')
sns.heatmap(df.corr(),annot=True)
plt.show()


DisplayOutliers(df,'Before Removing outliers')
df=RemoveOutlier(df,'Age')
df=RemoveOutlier(df,'EstimatedSalary')
DisplayOutliers(df,'After removing outliers:')

x=df[['Age','EstimatedSalary']]
y=df['Purchased']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression().fit(x_train,y_train)
y_pred=model.predict(x_test)

#display classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#display confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print('confusion_matrix:',cm)
fig,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.3,cmap='Blues')
plt.show()



