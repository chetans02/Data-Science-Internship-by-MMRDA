import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

df1 = pd.read_csv("walmart/train.csv")
reg = LinearRegression()
print(df1)

x = df1.drop('Weekly_Sales',axis=1)
y = df1['Weekly_Sales']

le = preprocessing.LabelEncoder()
df1['Date']= le.fit_transform(df1['Date'])
x = df1

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

reg_train = reg.fit(x_train,y_train)
reg_pred = reg.predict(x_test)

print("LinearRegression:", mean_squared_error(y_test,reg_pred)



#Results:

'''
Store  Dept        Date  Weekly_Sales  IsHoliday
0           1     1  2010-02-05      24924.50      False
1           1     1  2010-02-12      46039.49       True
2           1     1  2010-02-19      41595.55      False
3           1     1  2010-02-26      19403.54      False
4           1     1  2010-03-05      21827.90      False
...       ...   ...         ...           ...        ...
421565     45    98  2012-09-28        508.37      False
421566     45    98  2012-10-05        628.10      False
421567     45    98  2012-10-12       1061.02      False
421568     45    98  2012-10-19        760.01      False
421569     45    98  2012-10-26       1076.80      False

[421570 rows x 5 columns]
LinearRegression: 1.1431365551085166e-20  '''