import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

df = pd.read_csv("adult.csv")
reg = LinearRegression()
print(df.info())


le = preprocessing.LabelEncoder()
df['workclass']= le.fit_transform(df['workclass'])
df['education']= le.fit_transform(df['education'])
df['marital.status']= le.fit_transform(df['marital.status'])
df['occupation']= le.fit_transform(df['occupation'])
df['relationship']= le.fit_transform(df['relationship'])
df['race']= le.fit_transform(df['race'])
df['sex']= le.fit_transform(df['sex'])
df['native.country']= le.fit_transform(df['native.country'])
df['income']= le.fit_transform(df['income'])


x = df.drop('income',axis=1)
y = df['income']


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)


print('mean square error: ',mean_squared_error(y_test,reg_pred))

#Results:
''' mean square error:  0.13523685043672745 '''