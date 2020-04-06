# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
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

lm= Lasso()
lasso_model= lm.fit(X_train, y_train)
print(lasso_model)

print(lasso_model.intercept_)   #b0 katsayısını verecektir
print(lasso_model.coef_)        #bağımsız değişkenlerin katsayısını verecektir


#farklı lambda değerlerine karşılık katsayılar
lasso= Lasso()
coefs= []

alphas= np.random.randint(0,100000,10)
alphas2= 10**np.linspace(10,-1,100)*0.5

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train,y_train)
    coefs.append(lasso.coef_)

ax= plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")

print(lasso_model)

print(lasso_model.predict(X_train)[0:5])
print(lasso_model.predict(X_test)[0:5])


y_pred= lasso_model.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


r2hata= r2_score(y_test,y_pred)
print(r2hata)


lasso_cv_model= LassoCV(alphas= alphas, cv=10, max_iter=100000).fit(X_train,y_train)
print(lasso_cv_model.alpha_)


lasso_tuned= Lasso(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)
y_pred= lasso_tuned.predict(X_test)
print(y_pred)

rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


print(pd.Series(lasso_tuned.coef_, index=X_train.columns))

