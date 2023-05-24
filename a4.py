import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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
	fig.suptitle(message)
	sns.boxplot(data=df,x='rm',ax=axes[0])
	sns.boxplot(data=df,x='lstat',ax=axes[1])
	fig.tight_layout()
	plt.show()

#read dataset
df=pd.read_csv('BostonHousing.csv')
print('BostonHousing Dataset is loaded successfully...')

#Display  infromation of dataset
# print('Information of Dataset:\n',df.info)
# print('Shape of Dataset:\n',df.shape)
# print('Columnsname:n',df.columns)
# print('Total elements in Dataset:',df.size)
print('Datatype of attributes(columns):',df.dtypes)
# print('First 5 rows:',df.head())
# print('Last 5 rows:',df.tail())


#find missing values
print('Total number of values in Dataset:',df.isna().sum())

#finding and removing outliers
print('Finding and Removing outliers:')
DisplayOutliers(df,'Before Removing outliers')
df=RemoveOutlier(df,'rm')
df=RemoveOutlier(df,'lstat')
DisplayOutliers(df,'After removing outliers:')


#find correlation matrix using heatmap
sns.heatmap(df.corr(),annot=True)
plt.show()

#split data into inputs and outputs
x=df[['rm','lstat']]
y=df['medv']
#import train test split 

from sklearn.model_selection import train_test_split

#training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#apply linear regression model on training data
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error
print('MAE:',mean_absolute_error(y_test,y_pred))
print('Model Score:',model.score(x_test,y_test))

print('Predict House Price by giving user input:')
features=np.array([[16,19]])
prediction=model.predict(features)
print('prediction:{}'.format(prediction))