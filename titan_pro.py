# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:45:54 2018

@author: Aashish Mehtoliya
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
sns.set()

#load the data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


#analysing and altering data
df_train_test = [df_train,df_test]

for dataset in df_train_test:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.'
           ,expand=False)
title_mapping={'Mr':0,'Miss':1,'Mrs':2,'Master':3,
               'Dr':3,'Rev':3,'Major':3,'Mile':3,'Col':3,
               'Ms':3,'Capt':3,'Jonkheer':3,'Lady':3,'Sir':3,
               'Don':3,'Mme':3,'Countless':3}
for dataset in df_train_test:
    dataset['Title']=dataset['Title'].map(title_mapping)
    
df_train['Title'].fillna(0,inplace=True)
df_test['Title'].fillna(1,inplace=True)
    
sex_mapping={'male':0,'female':1}
for dataset in df_train_test:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)
    
def bar_chart(feature):
    Survived=df_train[df_train['Survived']==1][feature].values
    Dead=df_train[df_train['Survived']==0][feature].values
    df = pd.DataFrame([Survived,Dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    
df_train['Age'].fillna(df_train.groupby('Title')['Age'].transform('median'),inplace=True)
df_test['Age'].fillna(df_test.groupby('Title')['Age'].transform('median'),inplace=True)    

for dataset in df_train_test:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26),'Age']=1
    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36),'Age']=2
    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62),'Age']=3
    dataset.loc[dataset['Age']>62,'Age']=4

for dataset in df_train_test:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    
embarked_mapping = {'S':0, 'C':1,'Q':2}
for dataset in df_train_test:
    dataset['Embarked']=dataset['Embarked'].map(embarked_mapping)

df_train['Fare'].fillna(df_train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
df_test['Fare'].fillna(df_test.groupby('Pclass')['Fare'].transform('median'),inplace=True)



for dataset in df_train_test:
    dataset.loc[dataset['Fare']<=16,'Fare']=0
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30),'Fare']=1
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare']=2
    dataset.loc[dataset['Fare']>100,'Fare']=4
    
for dataset in df_train_test:
    dataset['Cabin']=dataset['Cabin'].str[:1]
    
cabin_mapping={'A':0.5,'B':1.0,'C':1.5,'D':2,'E':2.5,'F':3,'G':3.5,'T':4}
for dataset in df_train_test:
    dataset['Cabin']= dataset['Cabin'].map(cabin_mapping)
    
df_train['Cabin'].fillna(df_train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
df_test['Cabin'].fillna(df_test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)


#feature engineering
df_train['FamilySize']=df_train['SibSp']+df_train['Parch']+1
df_test['FamilySize']=df_test['SibSp']+df_test['Parch']+1

family_mapping={1: 0, 2: 0.5, 3: 1, 4: 1.5, 5: 2, 6: 2.5, 7: 3.0, 8: 3.5, 9: 4, 10: 4.5, 11: 5 }

for dataset in df_train_test:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)
    
feature_drop=['Parch','SibSp','Ticket','PassengerId','Name']
df_train=df_train.drop(feature_drop,axis=1)
df_test=df_test.drop(feature_drop,axis=1)


x=df_train.iloc[:,[1,2,3,4,5,6,7,8]]
y=df_train.iloc[:,0]


knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(x,y)
y_pred=knn.predict(df_test.iloc[:,:].values)
orig= pd.read_csv('test.csv')
orig=orig.iloc[:,0:1]
orig["Survived"]=y_pred[0:]
orig.to_csv("Submit.csv",header=True,index=False)
ans=pd.read_csv("Submit.csv")


#prediction by decision tree
des_tree = DecisionTreeClassifier(criterion='gini')
des_tree.fit(x,y)

y_pred_1=des_tree.predict(df_test.iloc[:,:].values)
orig_1=pd.read_csv('test.csv')
orig_1=orig_1.iloc[:,0:1]
orig_1["Survived"]=y_pred_1[0:]
orig_1.to_csv("Submit_1.csv",header=True,index=False)
ans_1=pd.read_csv("Submit_1.csv")



LRR = LogisticRegression()
LRR.fit(x,y)

y_pred_2=LRR.predict(df_test.iloc[:,:].values)
orig_2=pd.read_csv('test.csv')
orig_2=orig_2.iloc[:,0:1]
orig_2["Survived"]=y_pred_2[0:]
orig_2.to_csv("Submit_2.csv",header=True,index=False)
ans_2=pd.read_csv("Submit_2.csv")


Rfc = RandomForestClassifier(n_estimators =8,random_state=0)
Rfc.fit(x, y)

y_pred_3=Rfc.predict(df_test.iloc[:,:].values)
orig_3=pd.read_csv('test.csv')
orig_3=orig_3.iloc[:,0:1]
orig_3["Survived"]=y_pred_3[0:]
orig_3.to_csv("Submit_3.csv",header=True,index=False)
ans_3=pd.read_csv("Submit_3.csv")

