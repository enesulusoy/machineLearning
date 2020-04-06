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


df= pd.read_csv("Hitters.csv")
df= df.dropna()
dms= pd.get_dummies(df[["League","Division","NewLeague"]])  #kategorik değişkenleri dumm değişkenlere çevirme işlemi OneHotEncoding yaklaşımı yapmış olduk


y= df["Salary"]     #bağımlı değişkeni atama işlemi
X_= df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')   #bağımlı değişkeni ve kategorik değişkenleri veri setinde kaldırıp X_ atama işlemi 
 
X= pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)    #oluşturmuş olduğumuz dumm değişkenleri ve bağımsız değişkenleri bir araya getirme işlemi
#yukarda yapılan işlemler kategorik değişkenleri dumm çevirerek veri setinde tutup diğer bağımsız değişkenlerle birleştirdik

#aşağıda eğitim ve deneme seti olarak ayrıştırma işlemi yaptık
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25,random_state=42)

print(df.head())
print(df.shape)
print(X_train.head())


#Model Oluşturma ve Tahmin Ettirme İşlemi
knr= KNeighborsRegressor()
knn_model= knr.fit(X_train,y_train)
print(knn_model)
print(knn_model.n_neighbors)
print(knn_model.metric)


y_pred= knn_model.predict(X_test)[0:5]
y_pred= knn_model.predict(X_test)

rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Model Tuning
RMSE= []
for k in range(10):
    k= k+1
    knr= KNeighborsRegressor(n_neighbors=k)
    knn_model= knr.fit(X_train,y_train)
    y_pred= knn_model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_test,y_pred))
    RMSE.append(rmse)
    print("k= ",k, "için RMSE değeri: ",rmse)
    
    
#GridSearchCV hiperparametleri öğrenmemizi sağlayan kütüphane
knn_params= {"n_neighbors": np.arange(1,30,1)}
knr= KNeighborsRegressor()
gs= GridSearchCV(knr,knn_params,cv=10)    
knn_cv_model= gs.fit(X_train,y_train)
print(knn_cv_model.best_params_)    
    
    
#Final Model
knn= KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned= knn.fit(X_train,y_train) 

y_pred= knn_tuned.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

   