

import pandas as pd
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#data loadind
df = pd.read_csv("students.csv")
df

#data explore 
print(f"Number of Rows: {df.shape[2]} \nNumber of Columns: {df.shape[3]}")

df.head(3)
df.tail(3)
df.describe()
df.info()
print("-- Attributes in Data --")
for pd in df.columns:
    print(pd)
df.nunique()    

print("-- Number of Null Values in Data --")
print(df.isnull().sum())
#data preprocesing

df = df.drop('date', axis=1)
df.head(3)
df.info()
df['street'].mode()[0]
def fillNaObjMode(cols):
    for i in cols:
        df[i] = df[i].fillna(df[i].mode()[0])

columns = ['street','city','statezip','country']
fillNaObjMode(columns)

def fillNaIntMode(cols):
    for i in cols:
        df[i] = df[i].fillna(df[i].mode()[0])

columns = ['bedrooms','bathrooms','floors','waterfront','view','yr_built']
fillNaIntMode(columns)

def fillNaFloat(cols):
    for i in cols:
        df[i] = df[i].fillna(df[i].mean())

columns = ['price','sqft_living','sqft_lot','sqft_above','sqft_basement']
fillNaFloat(columns)
df.info()
def convertFloatintoInt(cols):
    for i in cols:
        df[i] = df[i].astype('int64')

columns = ['bedrooms','bathrooms','floors','waterfront','view','yr_built','price','sqft_living','sqft_lot','sqft_above','sqft_basement']
convertFloatintoInt(columns)
df.info()

def dataEncoder(cols):
    for i in cols:
        dataLabelEncoder = LabelEncoder()
        df[i] = dataLabelEncoder.fit_transform(df[i])

columns = ['city','statezip']
dataEncoder(columns)
df.info()

df.head(3)

#train test split
trainData, testData = train_test_split(df, test_size=0.2, shuffle=False)

trainData.shape
testData.shape

train_x = trainData.iloc()[:, 1:]
test_x  = testData.iloc()[:, 1:]

train_y = trainData.iloc()[:, 0]
test_y  = testData.iloc()[:, 0]
train_x.head(3)
train_y.head(3)
test_x.head(3)
test_y.head(3)

#classifier application
model_svc = SVC()
model_svc.fit(train_x, train_y)

print(model_svc)
pickle.dump(model_svc, open('model_svc.pkl', 'wb'))

model_svc = pickle.load(open('model_svc.pkl', 'rb'))
model_predictions = model_svc.predict(test_x)
model_accuracy_score = accuracy_score(test_y, model_predictions)
print("-- Model Accuracy Score: ", end='')
print(round(model_accuracy_score,3))

testdata_predict = testData.copy(deep=True)
pd.options.mode.chained_assignment = None
testdata_predict['Prediction'] = model_predictions
model_accuracy_score = accuracy_score(testdata_predict['price'], testdata_predict['Prediction'])
print("-- Model Accuracy Score: ", end='')
print(round(model_accuracy_score,3))
