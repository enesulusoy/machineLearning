# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


df= pd.read_csv("USArrests.csv", index_col=0)
print(df.head())
print(df.isnull().sum())    #veri setinde eksik gözlem var mı
print(df.info())    #veri seti hakkında bilgi
print(df.describe().T)     #betimsel istatistikler

#df.hist(figsize=(10,10))    #dağımları grafik ile görmek


hc_complete= linkage(df, "complete")
hc_average= linkage(df, "average")


#Veri setine göre otomatik kümeleme yapma işlemi grafikte gösterilmiştir
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           leaf_font_size=10)
plt.show()

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()



#Grafike bakıldıktan sonra kendi isteğimiz üzerine küme sayısını belirtirek oluşturma işlemi
plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10, #küme sayısını belirtir
           show_contracted=True,    #her hümenin eleman sayısı
           leaf_font_size=10)
plt.show()