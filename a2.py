import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def DetectOutlier(df,var):
     Q1 = df[var].quantile(0.25)
     Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3 + 1.5 * IQR, Q1-1.5*IQR
    print("Highest allowed in variable :", var, high)
    print("lowest allowed in variable:", var, low)
    count= df[(df[var]>high) | (df[var]<low)][var].count()
    print("Total outliers in:",var,':', count)
    df=df[((df[var]>=low) & (df[var] <= high))]
    print('Outlier removed in',var)
    return df

def DisplayOutliers(df,message):
    fig,axes=plt.subplots(2,2)
    fig.suptitle(message)
    sns.boxplot(data=df,x='raisedhands',ax=axes[0,0])
    sns.boxplot(data=df,x='VisITedResources',ax=axes[0,1])
    sns.boxplot(data=df,x='AnnouncementsView',ax=axes[1,0])
    sns.boxplot(data=df,x='Discussion',ax=axes[1,1])
    fig.tight_layout()
    plt.show()

 #read dataset
df=pd.read_csv('Academic_performance.csv')
print(df.columns)

#handling outliers
DisplayOutliers(df,'Before Removing outliers')
df=RemoveOutlier(df,'raisedhands')
df=RemoveOutlier(df,'VisITedResources')
df=RemoveOutlier(df,'AnnouncementsView')
df=RemoveOutlier(df,'Discussion')
DisplayOutliers(df,'After removing outliers')


sns.boxplot(data=df,x='gender',y='raisedhands',hue='gender')
plt.title('boxplot with 2 variables gender and raisedhands')
plt.show()

sns.boxplot(data=df,x='NationalITy',y='Discussion',hue='gender')
plt.title('boxplot with 3 variables gender,NationalITy,Discussion')
plt.show()


print('Relationships between variables using satterplot:')
sns.scatterplot(data=df,x='raisedhands',y='VisitedResources')
plt.title('scatterplot for raisedhands,VisitedResources')
plt.show()