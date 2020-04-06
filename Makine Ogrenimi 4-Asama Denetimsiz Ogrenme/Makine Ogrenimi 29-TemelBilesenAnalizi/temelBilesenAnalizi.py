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


df= pd.read_csv("Hitters.csv")
df.dropna(inplace=True)     #eksik verileri kaldırma
df= df._get_numeric_data()  #veri setindeki sadece sayısal verileri seçiyor
print(df.head())


from sklearn.preprocessing import StandardScaler
df= StandardScaler().fit_transform(df)
print(df[0:5,0:5])


from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca_fit= pca.fit_transform(df)

bilesen_df= pd.DataFrame(data= pca_fit,
                         columns= ["birinci_bilesen","ikinci_bilesen"])
print(bilesen_df)
print(pca.explained_variance_ratio_)
print(pca.components_[1])


#Optimum Bileşen Sayısı
pca= PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


#Final Model
pca= PCA(n_components=3)
pca_fit= pca.fit_transform(df)

print(pca.explained_variance_ratio_)
