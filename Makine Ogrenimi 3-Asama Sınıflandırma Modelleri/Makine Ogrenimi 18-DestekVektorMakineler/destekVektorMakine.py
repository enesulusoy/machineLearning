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
svm= SVC(kernel="linear")   #default olarak "rbf"
svm_model= svm.fit(x_train,y_train)
print(svm_model)

y_pred= svm_model.predict(x_test)
ilkeltesthata= accuracy_score(y_test,y_pred)
print(ilkeltesthata)


#Model Tuning
#%%
#2-5 dakika arası sürebilir
svm_params= {"C": np.arange(1,10),
             "kernel": ["linear","rbf"]}
svm= SVC()
gs= GridSearchCV(svm,
                 svm_params,
                 cv=5,
                 n_jobs=-1,
                 verbose=2)
svm_cv_model= gs.fit(x_train,y_train)

print(svm_cv_model.best_score_) #en iyi skoru verecektir
#0.7839044652128765
print(svm_cv_model.best_params_)    #en iyi skoru veren algoritmanın parametrelerini verir
#{'C': 2, 'kernel': 'linear'}


#Final Model
#%%
svm= SVC(C=2,kernel="linear")
svm_tuned= svm.fit(x_train,y_train)

y_pred= svm_tuned.predict(x_test)
dogrulukoranı= accuracy_score(y_test,y_pred)
print(dogrulukoranı)

