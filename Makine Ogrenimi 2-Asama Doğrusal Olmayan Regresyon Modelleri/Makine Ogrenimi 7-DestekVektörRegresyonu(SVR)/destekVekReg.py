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
svr= SVR("linear") #"rbf" de yazılabilir
svr_model= svr.fit(X_train,y_train)
print(svr_model)

print(svr_model.predict(X_test)[0:5])
print(svr_model.intercept_)
print(svr_model.coef_)


#Test Hatası
y_pred= svr_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Model Tuning
svr= SVR("linear")
svr_params= {"C": [0.1,0.5,1,3]}
gs= GridSearchCV(svr_model,svr_params,cv=5, verbose=2)
svr_cv_model= gs.fit(X_train,y_train)
print(svr_cv_model)
print(svr_cv_model.best_params_)
#%%
#verbose çalışırken durumu raporlamasını sağlar
#n_jobs paramatresi işlemci gücünü max da kullanmamızı sağlar
gs= GridSearchCV(svr_model,svr_params,cv=5, verbose=2, n_jobs=-1)
svr_cv_model= gs.fit(X_train,y_train)
print(svr_cv_model)
print(svr_cv_model.best_params_)
#%%

svr= SVR("linear",C=0.5)
svr_tuned= svr.fit(X_train,y_train)
y_pred= svr_tuned.predict(X_test)
print(y_pred)

rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
#%%