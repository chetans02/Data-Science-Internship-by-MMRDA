import pandas as pd
df=pd.read_csv("IRIS.csv")

print(df) #allrows

print(df.head(10)) #first10row
print(df.tail(20)) #last20row
print(df.info())

print(df.columns.values) #diff parameters
print(df.describe()) #all info

print(df.loc[:,["petal_length", "species"]])
print (df[df['petal_width']>0.1])
print(df[45:75])



