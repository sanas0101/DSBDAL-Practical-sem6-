def DetectOutlier(df, var):
 Q1 = df[var].quantile(0.25)
 Q3 = df[var].quantile(0.75)
 IQR = Q3 - Q1
 high, low = Q3 + 1.5 * IQR, Q1 - 1.5 * IQR
 print("highest allowed in variable:", var, high)
 print("lowest allowed in variable:", var, low)
 count = df[(df[var] > high) | (df[var] < low)][var].count()
 print('Total outlier in:', var, ':', count)
 df = df[((df[var] >= low) & (df[var] <= high))]
 print('Outlier removed in', var)
 print('Outlier removed in', var)
 return df

def Display(y_pred, y_test):
 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(y_test, y_pred)
 print('Confusion_matrix\n', cm)
 sns.heatmap(cm, annot=True)
 plt.show()
 from sklearn.metrics import classification_report
 print(classification_report(y_test, y_pred))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv')
print('Iris dataset is successfully loaded....')

#first
print('1. Display information of dataset')

columns = ['sepal_ength', 'sepal_width', 'petal_length', 'petal_width']
groupbycolumnname = ['Variety']
print('information about the dataset:', df.info())
print(df.head().T)
print(df.columns)

#second
print('2. Find missing values')

print(df.isnull().sum())

#third
print('3. Detect and remove outliers')

Variety = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
species = ['Setosa', 'Versicolor', 'Virginica']
fig, axes = plt.subplots(2, 2)
fig.suptitle('Before removing outliers')
sns.boxplot(data=df, x='sepal_length', ax=axes[0, 0])
sns.boxplot(data=df, x='sepal_width', ax=axes[0, 1])
sns.boxplot(data=df, x='petal_length', ax=axes[1, 0])
sns.boxplot(data=df, x='petal_width', ax=axes[1, 1])
plt.show()
print('Identifying overall outliers in feature variables.......')
for var in Variety:
 df = DetectOutlier(df, var)
fig, axes = plt.subplots(2, 2)
fig.suptitle('After removing Outliers')
sns.boxplot(data=df, x='sepal_length', ax=axes[0, 0])
sns.boxplot(data=df, x='sepal_width', ax=axes[0, 1])
sns.boxplot(data=df, x='petal_length', ax=axes[1, 0])
sns.boxplot(data=df, x='petal_width', ax=axes[1, 1])
fig.tight_layout()
plt.show()


#forth
print('4. Encoding using label encoder')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('iris.csv')
print(df.head().T)
print("\nColumn Names:\n")
print(df.columns)
encode = LabelEncoder()
df.species = encode.fit_transform(df.species)
print(df.head(10))

#fifth
print('5. find correlation matrix')
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=True)
plt.show()

#sixth
print('6. Train and test the model using Bernoulli Naive Bayes')
x = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, 
test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('y_pred=', y_pred)
from sklearn.metrics import accuracy_score
print('Accuracy of the BernoulliNB model:')
print(accuracy_score(y_pred, y_test))
Display(y_pred, y_test)


#seventh
print('7. Train and test the model using Guassian Naive Bayes')
x = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, 
test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
from sklearn.naive_bayes import GaussianNB
classifer = GaussianNB()
classifer.fit(x_train, y_train)
y_pred = classifer.predict(x_test)
print('y_pred=', y_pred)
from sklearn.metrics import accuracy_score
print('Accuracy of the GaussianNB model:')
print(accuracy_score(y_test, y_pred))
Display(y_pred, y_test)

#eighth
print('8. Prediction on user input')
import numpy as np
from sklearn.svm import SVC
x = df.iloc[:, [0, 1, 2, 3]].values
y = df.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, 
test_size=0.25, random_state=0)
model = SVC()
svm = SVC()
model.fit(x_train, y_train)
pred = model.predict(x_test)
svm.fit(x_train, y_train)
features = np.array([[4.0, 2.0, 4.1, 0.2]])
print(np.array)
prediction = svm.predict(features)
print('Prediction: {}'.format(prediction))
