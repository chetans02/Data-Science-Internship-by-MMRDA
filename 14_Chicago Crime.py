import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

'''
#fromfbprophet import Prophet
df1 = pd.read_csv("Chicago Crime/Chicago_Crimes_2005_to_2007.csv", error_bad_lines=False)
df2 = pd.read_csv("Chicago Crime/Chicago_Crimes_2008_to_2011.csv", error_bad_lines=False)
df3 = pd.read_csv("Chicago Crime/Chicago_Crimes_2012_to_2017.csv", error_bad_lines=False)
df = pd.concat([df1, df2, df3], ignore_index=False, axis=0)  '''



df = pd.read_csv("Chicago Crime/Chicago_Crimes_2012_to_2017.csv")

df = df.dropna()
df = df.drop(columns=['Unnamed: 0', 'ID', 'Case Number', 'IUCR','X Coordinate', 'Y Coordinate','Location'], axis = 1)
df= df.drop_duplicates()

df['date2'] = pd.to_datetime(df['Date'])
df['Year'] = df['date2'].dt.year
df['Month'] = df['date2'].dt.month
df['Day'] = df['date2'].dt.day
df['Hour'] = df['date2'].dt.hour
df['Minute'] = df['date2'].dt.minute
df['Second'] = df['date2'].dt.second
df = df.drop(['Date'], axis=1)
df = df.drop(['date2'], axis=1)
df = df.drop(['Updated On'], axis=1)
df.head()

df = df.sample(n=100000)

Classes = df['Primary Type'].unique()

df.drop(df.index [df[ 'Primary Type' ] == 'PUBLIC INDECENCY' ] , inplace = True)
df.drop(df.index [df[ 'Primary Type' ] == 'NON-CRIMINAL (SUBJECT SPECIFIED)' ] , inplace = True)
df.drop(df.index [df[ 'Primary Type' ] == 'NON-CRIMINAL' ] , inplace = True)
df.drop(df.index [df[ 'Primary Type' ] == 'NON - CRIMINAL' ] , inplace = True)
df.drop(df.index [df[ 'Primary Type' ] == 'OBSCENITY' ] , inplace = True)
df.drop(df.index [df[ 'Primary Type' ] == 'CONCEALED CARRY LICENSE VIOLATION' ] , inplace = True)


df_condition = [(df['Primary Type'] == 'MOTOR VEHICLE THEFT'),
                (df['Primary Type'] == 'THEFT'),
                (df['Primary Type'] == 'ROBBERY'),
                (df['Primary Type'] == 'BURGLARY'),
                (df['Primary Type'] == 'ASSAULT'),
                (df['Primary Type'] == 'PROSTITUTION'),
                (df['Primary Type'] == 'BATTERY'),
                (df['Primary Type'] == 'CRIM SEXUAL ASSAULT'),
                (df['Primary Type'] == 'SEX OFFENSE'),
                (df['Primary Type'] == 'INTIMIDATION'),
                (df['Primary Type'] == 'STALKING'),
                (df['Primary Type'] == 'ARSON'),
                 (df['Primary Type'] == 'KIDNAPPING'),
                (df['Primary Type'] == 'OFFENSE INVOLVING CHILDREN'),
                (df['Primary Type'] =='PUBLIC PEACE VIOLATION'),
                (df['Primary Type'] == 'OTHER NARCOTIC VIOLATION'),
                 (df['Primary Type'] == 'NARCOTICS'),
                (df['Primary Type'] == 'LIQUOR LAW VIOLATION'),
                (df['Primary Type'] == 'CRIMINAL DAMAGE'),
                (df['Primary Type'] == 'HUMAN TRAFFICKING'),
                 (df['Primary Type'] == 'WEAPONS VIOLATION'),
                (df['Primary Type'] == 'INTERFERENCE WITH PUBLIC OFFICER'),
                (df['Primary Type'] == 'CRIMINAL TRESPASS'),
                 (df['Primary Type'] == 'HOMICIDE'),
                (df['Primary Type'] == 'DECEPTIVE PRACTICE'),
                (df['Primary Type'] == 'OTHER OFFENSE'),
                (df['Primary Type'] == 'GAMBLING'
                )
               ]
df_categ = ['THEFT', 'THEFT', 'THEFT','THEFT',
            'ASSAULT' , 'ASSAULT' , 'ASSAULT' , 'ASSAULT', 'ASSAULT','ASSAULT','ASSAULT','ASSAULT','ASSAULT','ASSAULT', 'ASSAULT',
            'NARCOTICS', 'NARCOTICS', 'NARCOTICS',
            'CRIMINAL DAMAGE','CRIMINAL DAMAGE','CRIMINAL DAMAGE', 'CRIMINAL DAMAGE', 'CRIMINAL DAMAGE',
            'OTHER OFFENSE','OTHER OFFENSE','OTHER OFFENSE', 'OTHER OFFENSE']
df['Type'] = np.select(df_condition , df_categ)

df.info()
#print(df)

#Numbers of Crime by Type
plt.figure(figsize=(8,10))
pd.read_csv("Chicago Crime/Chicago_Crimes_2012_to_2017.csv").groupby([pd.read_csv("Chicago Crime/Chicago_Crimes_2012_to_2017.csv")['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by type')
plt.ylabel('Crime Type')
plt.xlabel('Number of crimes')
plt.show()

#Numbers of Crime by Location
plt.figure(figsize=(10,12))
pd.read_csv("Chicago Crime/Chicago_Crimes_2012_to_2017.csv").groupby([pd.read_csv("Chicago Crime/Chicago_Crimes_2012_to_2017.csv")['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by Location')
plt.ylabel('Crime Location')
plt.xlabel('Number of crimes')
plt.show()

data_ead = df[df.Year != 2017].drop(
    [
        "Year",
        "Community Area",
        "Latitude",
        "Longitude",
        "Block",
        "Primary Type",
        "Description",
        "Location Description",
        "Beat",
        "District",
        "Ward",
        "Year",
        "Latitude",
        "Longitude",	

    ],
    axis=1,)




df['Longitude'] = pd.factorize(df["Longitude"])[0]
df['Longitude'].unique()
df['Latitude'] = pd.factorize(df["Latitude"])[0]
df['Latitude'].unique()


Target = 'Type'
print('Target: ', Target)

Features = ["Longitude", "Latitude", "Minute"]
print('Full Features: ', Features)


x, y = train_test_split(df,test_size = 0.2,train_size = 0.8, random_state= 3)

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test

print('Feature Set Used    : ', Features)
print('Target Class        : ', Target)
print('Training Set Size   : ', x.shape)
print('Test Set Size       : ', y.shape)



rf = RandomForestClassifier(n_estimators=150, max_depth = 15) # Number of trees

# Model Training
rf.fit(X=x1,y=x2)

# Prediction
result = rf.predict(y[Features])

ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("=== Random Forest Results ====")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)


knn_model = KNeighborsClassifier(n_neighbors=150)

# Model Training
knn_model.fit(X=x1, y=x2)

# Prediction
result = knn_model.predict(y[Features])

ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("=== K-Nearest Neighbors Results ===")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)


#Results:
'''
=== Random Forest Results ====
Accuracy    :  0.40607182154646393
Recall      :  0.40607182154646393
Precision   :  0.3550200745635685
F1 Score    :  0.40607182154646393
Confusion Matrix: 
[[2740   28   59   16 2554]
 [1288   41   28   19 1697]
 [1361   20   49   15  511]
 [ 804   24   13   33 1342]
 [1968   41   38   49 5256]]
 
=== K-Nearest Neighbors Results ===
Accuracy    :  0.36165849754926477
Recall      :  0.36165849754926477
Precision   :  0.2645454198042358
F1 Score    :  0.36165849754926477
Confusion Matrix: 
[[ 798    0    9    2 4588]
 [ 390    0    4    3 2676]
 [ 263    0   13    5 1675]
 [ 291    0    3    3 1919]
 [ 912    0    7   16 6417]] '''

#There is an Error while installing fbprophet module in pycharm so I take the data of only one csv.file i.e ("Chicago Crime/Chicago_Crimes_2012_to_2017.csv")...the results will be same or little differnt for other csv.files...
