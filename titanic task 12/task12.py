import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#load titanic data set

train_data = pd.read_csv('train.csv')    
 
print(train_data.head())
print(train_data.info())
print(train_data.isnull().sum())

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)



le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])

train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
                 axis=1, inplace=True)

X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#Apply Classifiers
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_valid)
print("Logistic Regression Accuracy:", accuracy_score(y_valid, lr_preds))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_valid)
print("Random Forest Accuracy:", accuracy_score(y_valid, rf_preds))

svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_valid)
print("SVM Accuracy:", accuracy_score(y_valid, svc_preds))

#
final_preds = (lr_preds + rf_preds + svc_preds) // 2
print("Ensemble Accuracy:", accuracy_score(y_valid, final_preds))



