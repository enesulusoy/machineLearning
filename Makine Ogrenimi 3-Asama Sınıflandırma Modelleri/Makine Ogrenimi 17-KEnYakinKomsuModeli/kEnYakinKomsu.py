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


#Model ve Tahmin
#%%
knn= KNeighborsClassifier()
knn_model= knn.fit(x_train,y_train)
print(knn_model)

y_pred= knn_model.predict(x_test).astype(int)   #astype ile float olarak oluşan y_pred arrayi inte çevirdik bu sayede y_test ile karşılaştırılması mümkün olacak
basari= accuracy_score(y_test,y_pred)   #eğer astype ile int yapmasaydık accuracy_score(y_test,y_pred.round()) şeklinde kullanmamız gerekecekti
print(basari)


#Model Tuning
#%%
knn= KNeighborsClassifier()
knn_params= {"n_neighbors": np.arange(1,50)}

gs= GridSearchCV(knn, knn_params,cv=10)
knn_cv_model= gs.fit(x_train,y_train)
print(knn_cv_model.best_score_)     #en iyi skoru veriri
#0.748637316561845
print(knn_cv_model.best_params_)    #en iyi skoru veren model parametresi best parametre
#{'n_neighbors': 11}


#Final Model
#%%
knn= KNeighborsClassifier(n_neighbors=11)
knn_tuned= knn.fit(x_train,y_train)
y_pred= knn_tuned.predict(x_test)
print(accuracy_score(y_test,y_pred))
#0.7316017316017316
