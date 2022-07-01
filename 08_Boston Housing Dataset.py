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


#RESUKTS:
'''
      CRIM    ZN  INDUS  CHAS    NOX  ...  TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  ...  296     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  ...  242     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  ...  242     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  ...  222     18.7  394.63   2.94  33.4

[4 rows x 14 columns]
<bound method NDFrame.describe of         CRIM    ZN  INDUS  CHAS    NOX  ...  TAX  PTRATIO       B  LSTAT  MEDV
0    0.00632  18.0   2.31   0.0  0.538  ...  296     15.3  396.90   4.98  24.0
1    0.02731   0.0   7.07   0.0  0.469  ...  242     17.8  396.90   9.14  21.6
2    0.02729   0.0   7.07   0.0  0.469  ...  242     17.8  392.83   4.03  34.7
3    0.03237   0.0   2.18   0.0  0.458  ...  222     18.7  394.63   2.94  33.4
4    0.06905   0.0   2.18   0.0  0.458  ...  222     18.7  396.90    NaN  36.2
..       ...   ...    ...   ...    ...  ...  ...      ...     ...    ...   ...
501  0.06263   0.0  11.93   0.0  0.573  ...  273     21.0  391.99    NaN  22.4
502  0.04527   0.0  11.93   0.0  0.573  ...  273     21.0  396.90   9.08  20.6
503  0.06076   0.0  11.93   0.0  0.573  ...  273     21.0  396.90   5.64  23.9
504  0.10959   0.0  11.93   0.0  0.573  ...  273     21.0  393.45   6.48  22.0
505  0.04741   0.0  11.93   0.0  0.573  ...  273     21.0  396.90   7.88  11.9

[506 rows x 14 columns]>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   INDUS   486 non-null    float64
 1   NOX     506 non-null    float64
 2   RM      506 non-null    float64
 3   AGE     486 non-null    float64
 4   DIS     506 non-null    float64
 5   RAD     506 non-null    int64  
 6   TAX     506 non-null    int64  
 7   B       506 non-null    float64
 8   LSTAT   486 non-null    float64
dtypes: float64(7), int64(2)
memory usage: 35.7 KB
None
     INDUS    NOX     RM        AGE     DIS  RAD  TAX       B      LSTAT
0     2.31  0.538  6.575  65.200000  4.0900    1  296  396.90   4.980000
1     7.07  0.469  6.421  78.900000  4.9671    2  242  396.90   9.140000
2     7.07  0.469  7.185  61.100000  4.9671    2  242  392.83   4.030000
3     2.18  0.458  6.998  45.800000  6.0622    3  222  394.63   2.940000
4     2.18  0.458  7.147  54.200000  6.0622    3  222  396.90  12.715432
..     ...    ...    ...        ...     ...  ...  ...     ...        ...
501  11.93  0.573  6.593  69.100000  2.4786    1  273  391.99  12.715432
502  11.93  0.573  6.120  76.700000  2.2875    1  273  396.90   9.080000
503  11.93  0.573  6.976  91.000000  2.1675    1  273  396.90   5.640000
504  11.93  0.573  6.794  89.300000  2.3889    1  273  393.45   6.480000
505  11.93  0.573  6.030  68.518519  2.5050    1  273  396.90   7.880000

[506 rows x 9 columns]
0      24.0
1      21.6
2      34.7
3      33.4
4      36.2
       ... 
501    22.4
502    20.6
503    23.9
504    22.0
505    11.9
Name: MEDV, Length: 506, dtype: float64
LinearRegression: 32.63458728404159  '''




