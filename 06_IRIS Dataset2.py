import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv("IRIS.csv")

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
nb=MultinomialNB()

x = df.drop("species",axis=1)
y = df["species"]
print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

lr_train = lr.fit(x_train,y_train)
rf_train = rf.fit(x_train, y_train)
gbm_train = gbm.fit(x_train,y_train)
dt_train = dt.fit(x_train,y_train)
sv_train = sv.fit(x_train,y_train)
nb_train = nb.fit(x_train,y_train)



lr_pred = lr.predict(x_test)
rf_pred = rf.predict(x_test)
gbm_pred = gbm.predict(x_test)
dt_pred = dt.predict(x_test)
sv_pred = sv.predict(x_test)
nb_pred = nb.predict(x_test)


print('LogisticRegression')
print(accuracy_score(y_test,lr_pred))

print('Random Forest')
print(accuracy_score(y_test,rf_pred))

print('GradientBoostingClassifier')
print(accuracy_score(y_test,gbm_pred))

print('DecisionTreeClassifier')
print(accuracy_score(y_test,dt_pred))
      
print('svm')
print(accuracy_score(y_test,sv_pred))

print('naive bayes')
print(accuracy_score(y_test,nb_pred))



'''
Results:
1)LogisticRegression
0.9777777777777777

2)Random Forest
0.9777777777777777

3)GradientBoostingClassifier
0.9777777777777777

4)DecisionTreeClassifier
0.9777777777777777

5)svm
0.9777777777777777

6)naive bayes
0.6'''









