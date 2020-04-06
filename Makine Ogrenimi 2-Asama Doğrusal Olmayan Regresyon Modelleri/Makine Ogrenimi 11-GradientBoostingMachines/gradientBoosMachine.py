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
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                   random_state=42)

print(df.head())
print(df.shape)
print(X_train.head())


#Model ve Tahmin
gbm= GradientBoostingRegressor()
gbm_model= gbm.fit(X_train,y_train)
print(gbm_model)

y_pred= gbm_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Model Tuning
#%%
#bu işlem 3-5 dakika arası sürebilir
gbm_params= {"learning_rate": [0.001,0.1,0.01],
             "max_depth": [3,5,8],
             "n_estimators": [100,200,500],
             "subsample": [1,0.5,0.8],
             "loss": ["ls","lad","quantile"]}

gbm= GradientBoostingRegressor()
gbm_model= gbm.fit(X_train,y_train)

gs= GridSearchCV(gbm_model,
                 gbm_params,
                 cv=10,
                 n_jobs=-1,
                 verbose=2)

gbm_cv_model= gs.fit(X_train,y_train)
print(gbm_cv_model.best_params_)    #burada en iyi sonucu veren algoritmanın parametrelerinin değerlerini geri döndürüyor


#best parametlerinin sonucu:
#%%
#{'learning_rate': 0.1, 'loss': 'ls', 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.5}

gbm= GradientBoostingRegressor(learning_rate=0.1,
                               loss="ls",
                               max_depth=5,
                               n_estimators=200,
                               subsample=0.5)
gbm_tuned= gbm.fit(X_train,y_train)

y_pred= gbm_tuned.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Değişkenlerin Önem Düzeyleri
#%%
print(gbm_tuned.feature_importances_*100)    #veri setindeki değişkenlerin önemlerine göre puanlama yapar

Importance= pd.DataFrame({'Importance':gbm_tuned.feature_importances_*100},
                         index=X_train.columns)

Importance.sort_values(by='Importance',
                       axis=0,
                       ascending= True).plot(kind='barh',
                                             color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_=None    





