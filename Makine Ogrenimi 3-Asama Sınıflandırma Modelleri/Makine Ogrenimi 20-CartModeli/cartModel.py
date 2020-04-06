# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


df= pd.read_csv("diabetes.csv")
print(df.head())
y= df["Outcome"]
x= df.drop(['Outcome'], axis=1)
x_train, x_test, y_train, y_test= train_test_split(x,
                                                   y,
                                                   test_size=0.30,
                                                   random_state=42)


dtc= DecisionTreeClassifier()
cart_model= dtc.fit(x_train,y_train)
print(cart_model)

y_pred= cart_model.predict(x_test)
dogruluk= accuracy_score(y_test,y_pred)
print(dogruluk)


#Model Tuning
#%%
cart_params= {"max_depth": [1,3,5,8,10],
              "min_samples_split": [2,3,5,10,20,50]}

gs= GridSearchCV(dtc,
                 cart_params,
                 cv=10,
                 n_jobs=-1,
                 verbose=2)
cart_cv_model= gs.fit(x_train,y_train)
print(cart_cv_model)
print(cart_cv_model.best_params_)   #best parametre değerleri
#{'max_depth': 5, 'min_samples_split': 20}


#Final Model
#%%
dtc= DecisionTreeClassifier(max_depth=5,
                            min_samples_split=20) 

cart_tuned= dtc.fit(x_train,y_train)
y_pred= cart_tuned.predict(x_test)
dogruluk= accuracy_score(y_test,y_pred)
print(dogruluk)



