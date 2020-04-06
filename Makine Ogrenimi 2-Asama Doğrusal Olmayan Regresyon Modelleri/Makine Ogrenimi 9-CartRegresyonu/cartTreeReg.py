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


X_train= pd.DataFrame(X_train["Hits"])
X_test= pd.DataFrame(X_test["Hits"])


#Model ve Tahmin
dtr= DecisionTreeRegressor(max_leaf_nodes=10)
cart_model= dtr.fit(X_train,y_train)
print(cart_model)


X_grid= np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)
X_grid= X_grid.reshape((len(X_grid), 1))
                       
plt.scatter(X_train, y_train, color='red')

plt.plot(X_grid,cart_model.predict(X_grid),color='blue')

plt.title('CART REGRESYON AĞACI')
plt.xlabel('Atış Sayısı (Hits)')
plt.ylabel('Maaş (Salary)')
plt.show()                      


#Tek Değişkenli Tahmin
print(cart_model.predict(X_test)[0:5])

y_pred= cart_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Tüm Değişkenleri Kullanarak Tahmin
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

dtr= DecisionTreeRegressor()
cart_model= dtr.fit(X_train,y_train)

y_pred= cart_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


#Model Tuning
dtr= DecisionTreeRegressor(max_depth=5) #max_depth parametresi sayesinde modelin ne kadar derinleşeceğini belirtmek için kullanılırız
cart_model= dtr.fit(X_train,y_train)

y_pred= cart_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

cart_params= {"max_depth":[2,3,4,5,10,20],
              "min_samples_split":[2,10,5,30,50,10]}

#ağac modelinin içi boş olması için tekrar oluşturduk
cart_model= DecisionTreeRegressor() #max_depth parametresi sayesinde modelin ne kadar derinleşeceğini belirtmek için kullanılırız
#cart_model= dtr.fit(X_train,y_train)

gs= GridSearchCV(cart_model,cart_params,cv=10)
cart_cv_model= gs.fit(X_train,y_train)
print(cart_cv_model.best_params_)


#Final Model
dtr= DecisionTreeRegressor(max_depth=4, min_samples_split=50)
cart_tuned= dtr.fit(X_train,y_train)
print(cart_tuned)

y_pred= cart_tuned.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)