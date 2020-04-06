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
rf= RandomForestClassifier()
rf_model= rf.fit(x_train,y_train)
print(rf_model)

y_pred= rf_model.predict(x_test)
dogruluk= accuracy_score(y_test,y_pred)
print(dogruluk)


#Model Tuning
#%%
#2-5 dakika arası sürebilir
rf= RandomForestClassifier()
rf_params= {"n_estimators": [100,200,500,1000],#kullanılacak olan ağaç sayısı
            "max_features": [3,5,7,8],#max değişken sayısı
            "min_samples_split": [2,5,10,20]}

gs= GridSearchCV(rf,
                 rf_params,
                 cv=10,
                 n_jobs=-1,
                 verbose=2)
rf_cv_model= gs.fit(x_train,y_train)
print(rf_cv_model.best_params_) #best parametre değerleri
#{'max_features': 7, 'min_samples_split': 5, 'n_estimators': 500}


#Final Model
#%%
rf= RandomForestClassifier(max_features=7,
                           min_samples_split=5,
                           n_estimators=500)
rf_tuned= rf.fit(x_train,y_train)
y_pred= rf_tuned.predict(x_test)
dogruluk= accuracy_score(y_test,y_pred)
print(dogruluk)


#Değişken Önem Düzeyleri
#%%
print(rf_tuned.feature_importances_)    #değişken önem sırası
feature_imp= pd.Series(rf_tuned.feature_importances_,
                       index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri Random Forests")
plt.show()