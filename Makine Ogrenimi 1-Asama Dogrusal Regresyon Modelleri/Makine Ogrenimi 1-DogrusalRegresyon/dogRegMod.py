# -*- coding: utf-8 -*-

import pandas as pd

df= pd.read_csv("Advertising.csv")  #analiz yapılacak veri setinin okunması
df= df.iloc[:,1:len(df)]            #veri setinde istenen öğeleri alma

print(df.head())    #aldığımız verinin ilk satırlarını gösterme
print(df.info())    #çektiğimiz verinin içeriği hakkında bilgi edinmek

import seaborn as sns

print(sns.jointplot(x="TV",y="sales",data=df,kind="reg"))   #çekilen verinin görselleştirme işlemi

from sklearn.linear_model import LinearRegression

X= df[["TV"]]   #bagımsız değişken X ile ifade edeceğiz
print(X.head()) #bagımsız değişkenin ilk satırlarını gösterme

y= df[["sales"]]    #bagımlı değişken y
print(y.head())     #bagımlı değişkenin ilk satırlarını gösterme

reg= LinearRegression() #lineer regresyon modeli nesnesi oluşturma
model= reg.fit(X,y)     #oluşturulan model ile verinin eğitilme işlemi
print(model)            #oluşturulan modelin gösterilmesi

print(model.intercept_) #sabit sayı b0 (y=b0 + b1*x1)
print(model.coef_)      #katsayı b1
print(model.score(X,y)) #rkare ifadesi model skorunu gösterir

import matplotlib.pyplot as plt
g= sns.regplot(df["TV"], df["sales"], ci=None, scatter_kws={'color':'r', 's':9})    #veri görselleştirme işlemi
g.set_title("Model Denklemi: Sales= 7.03 + TV*0.05")    #görselleştirilene veriye başlık ekleme
g.set_ylabel("Satış Sayısı")        #görselleştirilen verinin y doğrusuna isim verme
g.set_xlabel("TV Harcamaları")      #görselleştirilen verinin x doğrusuna isim verme
plt.xlim(-10,310)
plt.ylim(bottom=0)

print(model.predict([[165]]))       #eğitilen model için verilen değerin tahminini yapmasını sağlama

yeni_veri= [[5],[15],[30]]          #yeni veriler oluşturma  
print(model.predict(yeni_veri))     #yeni verilerin modele göre tahmin ettirilmesi

gercek_y= y[0:10]       #veri setinden çekilen gerçek değerleri ayırma
tahmin_y= pd.DataFrame(model.predict(X)[0:10])      #model ile tahmin edilen değerlerin dataframe dönüştürülmesi

hatalar= pd.concat([gercek_y,tahmin_y],axis=1)      #ayrıştırdığımız tahmin ve gerçek değerlerin bir araya getirilmesi 
hatalar.columns= ["Gercek_y","Tahmin_y"]            #birleştirdiğimiz verilerin sütun başlıklarını değiştirme
print(hatalar)  #verilerin yazdırılması

hatalar["hata"]= hatalar.Gercek_y - hatalar.Tahmin_y    #modelin tahmin sonuçlarının gerçek değerler ile arasındaki fark
print(hatalar)

hatalar["hata_kareler"]= hatalar.hata**2        #hata kareler yöntemine göre hata oranının verilere eklenmesi
print(hatalar)

import numpy as np
print(np.mean(hatalar["hata_kareler"]))         #hata kareler ortalaması bulma işlemi



