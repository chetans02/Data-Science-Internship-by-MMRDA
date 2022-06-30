import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("winequalityN.csv")
reg = LinearRegression()

x = df.drop('type',axis=1)
x = x.drop('quality',axis=1)
x = x.drop('alcohol',axis=1)
x['fixed acidity'].fillna((x['fixed acidity'].mean()), inplace=True)
x['volatile acidity'].fillna((x['volatile acidity'].mean()), inplace=True)
x['citric acid'].fillna((x['citric acid'].mean()), inplace=True)
x['residual sugar'].fillna((x['residual sugar'].mean()), inplace=True)
x['chlorides'].fillna((x['chlorides'].mean()), inplace=True)
x['pH'].fillna((x['pH'].mean()), inplace=True)
x['sulphates'].fillna((x['sulphates'].mean()), inplace=True)

y = df['type']

#print(y)
le=LabelEncoder()
le.fit(y)
y = le.transform(y)


#print(x.info())
#print(y.info())

# # print(x)
# # print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print(mean_squared_error(y_test,reg_pred))

#Results:
''' 0.0318157979500277  '''