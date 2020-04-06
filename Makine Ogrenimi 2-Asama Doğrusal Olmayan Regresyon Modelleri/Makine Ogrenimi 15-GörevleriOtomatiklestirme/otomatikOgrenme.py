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
from catboost import CatBoostRegressor


df= pd.read_csv("Hitters.csv")
df= df.dropna()
dms= pd.get_dummies(df[["League","Division","NewLeague"]])  #kategorik değişkenleri dumm değişkenlere çevirme işlemi OneHotEncoding yaklaşımı yapmış olduk


def compML(df, y, alg):
    """
        df: işlem yapılacak veri seti
        y: bağımlı değişken
        alg: algoritma adı
    """
    #Train-test ayrimi:
    y= df[y]     #bağımlı değişkeni atama işlemi
    X_= df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')   #bağımlı değişkeni ve kategorik değişkenleri veri setinde kaldırıp X_ atama işlemi 
 
    #aşağıda yapılan işlemler kategorik değişkenleri dumm çevirerek veri setinde tutup diğer bağımsız değişkenlerle birleştirdik
    X= pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)    #oluşturmuş olduğumuz dumm değişkenleri ve bağımsız değişkenleri bir araya getirme işlemi

    #aşağıda eğitim ve deneme seti olarak ayrıştırma işlemi yaptık
    X_train, X_test, y_train, y_test= train_test_split(X,
                                                       y,
                                                       test_size=0.25,
                                                       random_state=42)

    #Modelleme İşlemi:
    model= alg().fit(X_train,y_train)
    y_pred= model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_test,y_pred))
    return rmse
    

def compMLA(df, y, alg):
    """
        df: işlem yapılacak veri seti
        y: bağımlı değişken
        alg: algoritma adı
    """
    #Train-test ayrimi:
    y= df[y]     #bağımlı değişkeni atama işlemi
    X_= df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')   #bağımlı değişkeni ve kategorik değişkenleri veri setinde kaldırıp X_ atama işlemi 
 
    #aşağıda yapılan işlemler kategorik değişkenleri dumm çevirerek veri setinde tutup diğer bağımsız değişkenlerle birleştirdik
    X= pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)    #oluşturmuş olduğumuz dumm değişkenleri ve bağımsız değişkenleri bir araya getirme işlemi

    #aşağıda eğitim ve deneme seti olarak ayrıştırma işlemi yaptık
    X_train, X_test, y_train, y_test= train_test_split(X,
                                                       y,
                                                       test_size=0.25,
                                                       random_state=42)

    #Modelleme İşlemi:
    model= alg().fit(X_train,y_train)
    y_pred= model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_test,y_pred))
    model_ismi= alg.__name__
    print(model_ismi," Modeli Test Hatası: ",rmse)

hatalgbm= compML(df, "Salary", LGBMRegressor)
print("hatalightgbm: ",hatalgbm)

hatasvr= compML(df, "Salary", SVR)
print("hatasvr: ",hatasvr)

hataxcb= compML(df, "Salary", XGBRegressor)
print("hataxcb: ",hataxcb)

hatarfr= compML(df, "Salary", RandomForestRegressor)
print("hatarfr: ",hatarfr)

hataknr= compML(df, "Salary", KNeighborsRegressor)
print("hataknr: ",hataknr)

hatadtr= compML(df, "Salary", DecisionTreeRegressor)
print("hatadtr: ",hatadtr)

hatagbr= compML(df, "Salary", GradientBoostingRegressor)
print("hatagbr: ",hatagbr)

hatalr= compML(df, "Salary", LinearRegression)
print("hatalr: ",hatalr)

#%%
models= [LGBMRegressor,
         XGBRegressor,
         GradientBoostingRegressor,
         RandomForestRegressor,
         DecisionTreeRegressor,
         MLPRegressor,
         KNeighborsRegressor,
         SVR]

for i in models:
    
    compMLA(df, "Salary", i)
    
    
    
    
    
    