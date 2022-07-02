import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import collections
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


df=pd.read_csv('Black Friday Sales/train.csv')

df.head(3)
d_c = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3', 'Purchase']

df.drop(['User_ID','Product_ID'], axis=1, inplace=True)
df.isnull().sum()
df.isnull().mean()

#The Unique element present in Gender
print('The Unique element present in Gender')
print(df['Gender'].unique())
sns.countplot(df['Gender'])
plt.show()
c=collections.Counter(df['Gender'])
print('The count of Each unique element')
print(c)


#The Unique element present in Age'
print('The Unique element present in Age')
print(df['Age'].unique())
plt.figure(figsize=(10,5))
sns.countplot(df['Age'])
plt.show()
print('The count of Each unique element')
print(df['Age'].value_counts())
df['Age'].value_counts(normalize = True).plot.bar(title = 'Age')


#Product_Category_1
print('The Unique element present in Product_Category_1')
print(df['Product_Category_1'].unique())
plt.figure(figsize=(10,5))
sns.countplot(df['Product_Category_1'])
plt.show()
print('The count of Each unique element')
print(df['Product_Category_1'].value_counts())
plt.figure(figsize=(10,5))
df['Product_Category_1'].value_counts(normalize = True).plot.bar(title = 'Product_Category_1')


df['Product_Category_3'].isnull().mean()
df.drop(['Product_Category_3'], axis=1, inplace=True)


df1=df.copy()
df1.dropna(inplace=True)
df1.isnull().mean()

print(df1.dtypes)
df1['Occupation']=df1['Occupation'].astype('object')
df1['Occupation'].dtypes, df1['Age'].dtypes
#print(df1.dtypes)

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df1['Gender']=l.fit_transform(df1['Gender'])
df1['Age']=l.fit_transform(df1['Age'])
df1['Occupation']=l.fit_transform(df1['Occupation'])
df1['City_Category']=l.fit_transform(df1['City_Category'])
df1['Stay_In_Current_City_Years']=l.fit_transform(df1['Stay_In_Current_City_Years'])
df1.head()
#print(df1.head())

df1['Occupation'].unique()
df1['Product_Category_2']=df1['Product_Category_2'].astype('int64')
df1.head(1)
#print(df1.head(1))


X=df1.drop(['Purchase'], axis=True)
y=df1['Purchase']

from sklearn.preprocessing import MinMaxScaler
df_num_scl=MinMaxScaler()
x=df_num_scl.fit_transform(X)
x=pd.DataFrame(data=x, columns=X.columns)

print(x[:1])
X_d=x[['Product_Category_1']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model=lr.fit(X_train, y_train)

print("LinearRegression:", model.score(X_test,y_test))

#Results:
''' LinearRegression: 0.17770530608938273 '''


