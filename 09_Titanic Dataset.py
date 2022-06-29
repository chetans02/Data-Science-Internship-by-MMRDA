
#1)LogisticRegression:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("tested.csv")
bos = load_boston
lr = LogisticRegression ()

# print(df.head(4))
# print(df.describe())

x = df.drop('PassengerId',axis=1)
x = x.drop('Survived', axis=1)
x = x.drop('Name',axis=1)
x = x.drop('Ticket',axis=1)
x = x.drop('Cabin',axis=1)
x = x.drop('Embarked',axis=1)
x = x.drop('Parch',axis=1)
x = x.drop('Sex',axis=1)

y = df['Survived']

x['Age'].fillna((x['Age'].max()), inplace=True)
x['Fare'].fillna((x['Fare'].mean()), inplace=True)
# print(x.info())
# print(y.info())

# print(x)
# print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

lr_train = lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)


print('LogisticRegression')
print(accuracy_score(y_test,lr_pred))





#2)LinearRegression:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("tested.csv")
bos = load_boston
reg = LinearRegression ()

# print(df.head(4))
# print(df.describe())

x = df.drop('PassengerId',axis=1)
x = x.drop('Survived', axis=1)
x = x.drop('Name',axis=1)
x = x.drop('Ticket',axis=1)
x = x.drop('Cabin',axis=1)
x = x.drop('Embarked',axis=1)
x = x.drop('Parch',axis=1)
x = x.drop('Sex',axis=1)

y = df['Survived']

x['Age'].fillna((x['Age'].mean()), inplace=True)
x['Fare'].fillna((x['Fare'].mean()), inplace=True)
# print(x.info())
# print(y.info())

#print(x)
#print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print('LinearRegression')
print(mean_squared_error(y_test,reg_pred))

#Results
'''
LogisticRegression
0.5396825396825397
LinearRegression
0.24553406029157332    '''

