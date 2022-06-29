import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("HousingData.csv")
bos = load_boston
reg = LinearRegression ()

print(df.head(4))
print(df.describe)


x = df.loc[:, 'LSTAT']
y = df['MEDV']
'''
print(X)
print(Y)
plt.scatter(X,Y)
plt.show() '''

x = df.drop('CRIM',axis=1)
x = x.drop('ZN', axis=1)
x = x.drop('CHAS',axis=1)
x = x.drop('PTRATIO',axis=1)
x = x.drop('MEDV', axis=1)
print(x.info())

x['INDUS'].fillna((x['INDUS'].max()), inplace=True)
x['AGE'].fillna((x['AGE'].mean()), inplace=True)
x['LSTAT'].fillna((x['LSTAT'].mean()), inplace=True)

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print(mean_squared_error(y_test,reg_pred))





