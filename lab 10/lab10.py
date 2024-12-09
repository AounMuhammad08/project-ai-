
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
df = pd.read_csv("data.csv")

# Handle missing data
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
df = pd.get_dummies(df, columns=['street', 'city', 'statezip', 'country'], drop_first=True)

scaler = StandardScaler()
numeric_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'price']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}
svc = SVC()
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_svc = grid_search.best_estimator_
y_pred = best_svc.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
with open('best_svc_model.pkl', 'wb') as f:

    pickle.dump(best_svc, f)



