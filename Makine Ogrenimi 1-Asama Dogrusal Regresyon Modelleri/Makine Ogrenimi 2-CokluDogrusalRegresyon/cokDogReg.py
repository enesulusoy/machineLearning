# -*- coding: utf-8 -*-

import pandas as pd

df= pd.read_csv("Reklam.csv")
df= df.iloc[:,1:len(df)]
print(df.head())


X= df.drop('sales',axis=1)  #bağımsız değişkenleri X olarak ayırdık
y= df["sales"]  #bağımlı değişkenleri y olarak ayırdık
print(y.head())
print(X.head())

y= df[["sales"]]    #dataframe çevirmek için çift parantez
print(y.head())


#Statsmodel ile model kurmak
import statsmodels.api as sm
lm= sm.OLS(y,X)
model= lm.fit()
print(model.summary())

"""
#scikit learn ile model kurmak
from sklearn.linear_model import LinearRegression
lm= LinearRegression()
model= lm.fit(X,y)
print(model.intercept_) #sabit sayıyı verir b0
#Çıktı=> [2.93888937]
print(model.coef_)  #bagımsız değişkenlerin katsayılarını verir
#Çıktı=> [[ 0.04576465  0.18853002 -0.00103749]]

#--verilen katsayılara göre dogru denklemi formu:
#--sales= 2.94 + TV*0.04 + radio*0.19 - newspaper*0.001


#30 birim TV, 10 birim radio, 40 birim gazete olursa sonuc ne olur
print(2.94+30*0.04+10*0.19-40*0.001)

yeni_veri=[[30],[10],[40]]  #istenilen değerleri tahmin ettirmek için veri oluşturduk
yeni_veri= pd.DataFrame(yeni_veri).T    #oluşturduğumuz veriyi dataframe e cevirme işlemi
print(yeni_veri)

print(model.predict(yeni_veri)) #oluşturduğumuz veriyi kullanarak modelimizin bu verilere göre sonucunun ne olacağını tahmin ettirilmesi


#model başarısı değerlendirmek
from sklearn.metrics import  mean_squared_error  
MSE= mean_squared_error(y,model.predict(X))  #modelin tahminlerine göre hata kareler ortalaması sonucunu verir MSE
print(MSE)    

import numpy as np
RMSE= np.sqrt(MSE)  #modelin tahminlerine göre hata kareler karekökü ortalaması sonucunu verir RMSE
print(RMSE)

#------------------------------------------------
#Model Tuning (Model Doğrulama)
#sinama seti yaklaşımı ile hata hesaplama
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.20,random_state=99)
print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

lm= LinearRegression()
model= lm.fit(X_train,y_train)

#eğitim hatası
hataegitim= np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
print(hataegitim)

#test hatasi
hatatest= np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
print(hatatest)


#k-katli cross validation (train test hatasını daha iyi değerlendirmek için gerekli bir yöntem)
from sklearn.model_selection import cross_val_score
hataoran= cross_val_score(model,X_train,y_train,cv=10,scoring="neg_mean_squared_error") 
print(hataoran)     #cv de verilen değer kadar hata hesaplaması yapar tüm parçaları tek tek dısarda bakıp deneyerek hata oranları çıkartır
print(np.mean(-hataoran))     #cross validation ile MSE
print(np.sqrt(np.mean(-hataoran)))   #cross v ile RMSE

"""


