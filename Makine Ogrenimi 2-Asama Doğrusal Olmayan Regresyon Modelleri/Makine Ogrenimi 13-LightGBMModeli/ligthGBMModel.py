# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from warnings import filterwarnings
filterwarnings('ignore')

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

df= pd.read_csv("Hitters.csv")
df= df.dropna()
dms= pd.get_dummies(df[["League","Division","NewLeague"]])  #kategorik değişkenleri dumm değişkenlere çevirme işlemi OneHotEncoding yaklaşımı yapmış olduk


y= df["Salary"]     #bağımlı değişkeni atama işlemi
X_= df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')   #bağımlı değişkeni ve kategorik değişkenleri veri setinde kaldırıp X_ atama işlemi 
 
X= pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)    #oluşturmuş olduğumuz dumm değişkenleri ve bağımsız değişkenleri bir araya getirme işlemi
#yukarda yapılan işlemler kategorik değişkenleri dumm çevirerek veri setinde tutup diğer bağımsız değişkenlerle birleştirdik

#aşağıda eğitim ve deneme seti olarak ayrıştırma işlemi yaptık
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                   random_state=42)

print(df.head())
print(df.shape)
print(X_train.head())


#Model ve Tahmin
#%%
lgb= LGBMRegressor()
lgb_model= lgb.fit(X_train,y_train)
print(lgb_model)

y_pred= lgb_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Model Tuning
#%%
lgb= LGBMRegressor()
lgb_params= {"learning_rate": [0.01,0.1,0.5,1],
             "n_estimators": [20,40,100,200,500,1000],
             "max_depth": [1,2,3,4,5,6,7,8,9,10]}

gs= GridSearchCV(lgb_model,
                 lgb_params,
                 cv=10,
                 n_jobs=-1,
                 verbose=2)

lgb_cv_model= gs.fit(X_train,y_train)
print(lgb_cv_model)
print(lgb_cv_model.best_params_)


#Final Model
#%%
lgb= LGBMRegressor(learning_rate=0.1,
                   max_depth=6,
                   n_estimators=20)
lgb_tuned= lgb.fit(X_train,y_train)
print(lgb_tuned)

y_pred= lgb_tuned.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


