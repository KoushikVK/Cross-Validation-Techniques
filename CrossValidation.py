# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 12:20:48 2021

Cross Validation

@author: Koushik V
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##we are going to predict which type of cancer it is
df = pd.read_csv('cancer_dataset.csv')
df= df.iloc[:,:-1]
df.drop('id',axis =1 ,inplace = True)

#Dependent and independent Variables
X= df.iloc[:,1:]
y = df.iloc[:,:1]
print(y.value_counts())


#Check if there is any null values
X.isnull().sum()

#HoldOut Validation Approach- Train And Test Split

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)


##kfold Cross validation 
from sklearn.model_selection import KFold
model = DecisionTreeClassifier()
kfold = KFold(10)  ##no of experiments

from sklearn.model_selection import cross_val_score
kfoldresults=cross_val_score(model,X,y,cv=kfold)
print(kfoldresults)
print(np.mean(kfoldresults))

# =============================================================================
# [0.87719298 0.9122807  0.89473684 0.92982456 0.9122807  0.98245614
#  0.89473684 0.96491228 0.96491228 0.94642857]
# 0.9279761904761905   ##Average Accuracy 
# =============================================================================

##stratified Cross validation 

from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold(n_splits=10)
model=DecisionTreeClassifier()
scores=cross_val_score(model,X,y,cv=skfold)
print(np.mean(scores))


##repeated train_test split   ##random train test split 

from sklearn.model_selection import ShuffleSplit
model=DecisionTreeClassifier()
ssplit=ShuffleSplit(n_splits=10,test_size=0.30)
shuffleresults=cross_val_score(model,X,y,cv=ssplit)


print(shuffleresults)

# =============================================================================
# [0.9005848  0.92982456 0.91812865 0.92982456 0.95321637 0.91812865
#  0.94152047 0.91812865 0.92397661 0.94736842]
# 
# 
# =============================================================================

print(np.mean(shuffleresults))

#0.9280701754385966







