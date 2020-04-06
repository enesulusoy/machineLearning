# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


df= pd.read_csv("Hitters.csv")
df= df.dropna() #veri seti içersinde eksik değer olduğunda kaldırma işlemi yapar
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


enet_model= ElasticNet().fit(X_train,y_train)
print(enet_model)
print(enet_model.coef_)
print(enet_model.intercept_)


#Tahmin
y_pred= enet_model.predict(X_train)[0:10]
print(y_pred)

y_pred= enet_model.predict(X_test)[0:10]
print(y_pred)

y_pred= enet_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(y_pred)
print(rmse)

print(r2_score(y_test,y_pred))


#Model Tuning
alphas= np.random.randint(0,100000,10)
alphas2= 10**(np.linspace(10,-2,100)*0.5)


enet_cv_model= ElasticNetCV(alphas=alphas2, cv=10).fit(X_train,y_train)
print(enet_cv_model.alpha_)
print(enet_cv_model.intercept_)
print(enet_cv_model.coef_)


#Final Model
enet= ElasticNet(alpha= enet_cv_model.alpha_)
enet_tuned= enet.fit(X_train,y_train)
y_pred= enet_tuned.predict(X_test)
print(enet_tuned)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)