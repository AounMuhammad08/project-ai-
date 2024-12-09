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