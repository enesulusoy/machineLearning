# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


df= pd.read_csv("USArrests.csv", index_col=0)
print(df.head())
print(df.isnull().sum())    #veri setinde eksik gözlem var mı
print(df.info())    #veri seti hakkında bilgi
print(df.describe().T)     #betimsel istatistikler

#df.hist(figsize=(10,10))    #dağımları grafik ile görmek

kmeans= KMeans(n_clusters=4)
k_fit= kmeans.fit(df)

print(k_fit.n_clusters) #class sayısı
print(k_fit.cluster_centers_) #merkez sayısı
print(k_fit.labels_) #gözlem birimlerinin hen class a ait oldukları verir

#Kümelerin Görselleştirilmesi
k_means= KMeans(n_clusters=2)
k_means_fit= k_means.fit(df)
kumeler= k_means_fit.labels_
print(kumeler)

plt.scatter(df.iloc[:,0],
            df.iloc[:,1],
            c=kumeler,
            s=50,
            cmap="viridis")

merkezler= k_means_fit.cluster_centers_
print(merkezler)

plt.scatter(merkezler[:,0],
            merkezler[:,1],
            c="black",
            s=200,
            alpha=0.5)
plt.show()

#Optimum Küme Sayısının Belirlenmesi
#Elbow Yöntemi
#%%
ssd= [] #uzaklık farklarının kareleri
K= range(1,30)

for k in K:
    kmeans= KMeans(n_clusters=k)
    kmeans_model= kmeans.fit(df)
    ssd.append(kmeans_model.inertia_)

plt.plot(K,ssd,"bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme Sayısı için Elbow Yöntemi")
plt.show()

#Optimum küme sayısını seçerken grafiğe bakılarak seçerken en büyük kırılma
#olan yere odaklanılır ve o kadar küme oluşturulması istenir. (örnek=3)

visu= KElbowVisualizer(kmeans,k=(2,20))
visu_fit= visu.fit(df)
visu_fit.poof()
#visu sayesinde grafikte en iyi küme seçilerek bize gösterilmesi sağlanır


#Final Model
kmeans= KMeans(n_clusters=4)
kmeans_model= kmeans.fit(df)
print(kmeans_model)


kumeler= kmeans_model.labels_
kume= pd.DataFrame({"Eyaletler":df.index, "Kumeler":kumeler})
print(kume)


df["Kume_No"]= kumeler
print(df)
