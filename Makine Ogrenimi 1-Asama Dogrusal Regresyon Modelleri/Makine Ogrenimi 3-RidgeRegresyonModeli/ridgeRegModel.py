# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection


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

#ridge model oluşturma işlemi
rm= Ridge(alpha=0.1)
ridge_model= rm.fit(X_train,y_train)
print(ridge_model)
print(ridge_model.coef_)
print(ridge_model.intercept_)


print(np.linspace(10,-2,100))   #rastgele sayılar üretme işlemi
lambdalar= 10**np.linspace(10,-2,100)*0.5
print(lambdalar)

ridge_model= Ridge()
katsayılar= []

#her lambda nın bağımsız değişken sayısı kadar katsayısı olacaktır
for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train,y_train)
    katsayılar.append(ridge_model.coef_)


ax= plt.gca()
ax.plot(lambdalar,katsayılar)
ax.set_xscale("log")

rm= Ridge()
ridge_model= rm.fit(X_train,y_train)
y_pred= ridge_model.predict(X_train)
print(y_pred[0:10])
print(y_train[1:10])


#Train Hatası-------------------
RMSE= np.sqrt(mean_squared_error(y_train,y_pred))   #gerçek değerler üzerinden tamin edilen değerlerin hata toplam karekökün ortalaması çıkarma
print(RMSE)

dogrulama= cross_val_score(ridge_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
HKO=np.mean(-dogrulama) #np.mean(-cross_val_score(ridge_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")) = hata karaler toplamı ortalamasını
print(HKO)

HKOK= np.sqrt(HKO)  #np.sqrt(np.mean(-dogrulama)) = hata karaler toplamı ortalamasının karekökü
print(HKOK)


#Test Hatası---------------------
y_pred= ridge_model.predict(X_test)

RMSE= np.sqrt(mean_squared_error(y_test,y_pred))   #gerçek değerler üzerinden tamin edilen değerlerin hata toplam karekökün ortalaması çıkarma
print(RMSE)


#Model Tuning---------
rm= Ridge()
ridge_model= rm.fit(X_train,y_train)
y_pred= ridge_model.predict(X_test)

MSE= np.sqrt(mean_squared_error(y_test, y_pred))
print(MSE)

lambdalar1= np.random.randint(0,1000,100)
lambdalar2= 10**np.linspace(10,-1,100)*0.5

ridgecv= RidgeCV(alphas= lambdalar2, scoring= "neg_mean_squared_error", cv=10, normalize=True)
ridgecv_model= ridgecv.fit(X_train,y_train)
print(ridgecv_model)


print(ridgecv_model.alpha_)


#Final Modeli Oluşturma
rmFinal= Ridge(alpha= ridgecv_model.alpha_)
ridge_tuned= rmFinal.fit(X_train,y_train)

y_pred= ridge_tuned.predict(X_test)

MSE= np.sqrt(mean_squared_error(y_test, y_pred))
print(MSE)



