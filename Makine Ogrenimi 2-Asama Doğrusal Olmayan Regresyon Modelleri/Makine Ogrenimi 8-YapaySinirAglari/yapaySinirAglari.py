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


#Model ve Tahmin
scaler= StandardScaler()
scaler.fit(X_train)
X_train_scaled= scaler.transform(X_train)
scaler.fit(X_test)
X_test_scaled= scaler.transform(X_test)


mlp= MLPRegressor()
mlp_model= mlp.fit(X_train_scaled,y_train)
print(mlp_model)

y_pred= mlp_model.predict(X_test_scaled)
print(y_pred)

rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

#%%
#Model Tuning
mlp_params= {"alpha": [0.1,0.01,0.02,0.001,0.0001],
             "hidden_layer_sizes": [(10,20),(5,5),(100,100)]}    
#"hidden_layer_sizes" değerleri içersinde 3 adet farklı layer bu layerler sütun 
#sayısı gizli katman sayısına denk geliyorken sütün değeri yani satırlar katmanların
#nöron sayısını belirtmek için yazılacaktır

#*******mlp_params ın amacı:*********
#amaç ilk alpha değeri ile ile ilk hidden layeri deneyecek sonra tek tek diğer
#gizli katmanlar ile denenecek ondan sonra sıradaki alpha değeri ile gizli
#katmanlar sırayla karşılaştırılmaya devam edecek


gs= GridSearchCV(mlp_model,mlp_params,cv=10,verbose=2,n_jobs=-1)
mlp_cv_model= gs.fit(X_train_scaled,y_train)
#print(mlp_cv_model)

print(mlp_cv_model.best_params_)    #best alpha değerini verir

#%%
#Final Model
mlp= MLPRegressor(alpha=0.1,hidden_layer_sizes=(100,100))    #alpha değeri ile gizli katman sayısı ve içerisndeki nöron sayısını yazıyoruz
mlp_tuned= mlp.fit(X_train_scaled,y_train)

y_pred= mlp_tuned.predict(X_test_scaled)
print(y_pred)

rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
#%%