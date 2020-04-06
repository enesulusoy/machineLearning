# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


df= pd.read_csv("diabetes.csv")
print(df.head())
y= df["Outcome"]
x= df.drop(['Outcome'], axis=1)
x_train, x_test, y_train, y_test= train_test_split(x,
                                                   y,
                                                   test_size=0.30,
                                                   random_state=42)

scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
scaler.fit(x_test)
x_test= scaler.transform(x_test)
#Model ve Tahmin
#%%
mlp= MLPClassifier()
mlp_model= mlp.fit(x_train,y_train)
print(mlp_model.coefs_) #bu sinir ağının kullanmış olduğu farklı katmanlardaki farklı
#hücrelerin birbirleriyle ilişkileri ve benzeri durumları ifade eden katsayıları verir

y_pred= mlp_model.predict(x_test)
dogruluk= accuracy_score(y_test,y_pred)
print(dogruluk)


#Model Tuning
#%%
#dogrusal problem için activation: "relu" içinde linear fonksiyon kullanılır
#sınıflandırma problemleri için activation: "logistic" içinde sigmoid kullanılır

#veri seti küçük olduğunda solver: "lbfgs"
#veri seti büyük olduğunda solver: "adam"

#alpha ceza terimidir onde göre değerler verilir optimizasyonu sağlamak için kullanılır

#hidden_layer_sizes: gizli katman sayısı ve hücre sayısını belirtmek için kullanılır
#solver: ağırlık optimizasyonu yapmak için kullanılır
#alpha: ceza terimi

mlp_params= {"alpha": [1,5,0.1,0.01,0.03,0.005,0.0001],
             "hidden_layer_sizes": [(10,10),(100,100,100),(100,100),(3,5)]}

mlp= MLPClassifier(solver="lbfgs",activation="logistic")
gs= GridSearchCV(mlp,
                 mlp_params,
                 cv=10,
                 n_jobs=-1,
                 verbose=2)
mlp_cv_model= gs.fit(x_train,y_train)
print(mlp_cv_model)
print(mlp_cv_model.best_params_)    #en iyi parametre değerleri
#activation değeri yokken sonuc=  {'alpha': 0.0001, 'hidden_layer_sizes': (100, 100, 100)}
#activation değeri "logistic" o=  {'alpha': 5, 'hidden_layer_sizes': (100, 100)}
#scaler işlemine tabi tutuldu s=  {'alpha': 1, 'hidden_layer_sizes': (3, 5)}

#Final Model
#%%
mlp= MLPClassifier(solver="lbfgs",
                   activation="logistic",
                   alpha=1,
                   hidden_layer_sizes=(3,5))

mlp_tuned= mlp.fit(x_train,y_train)

y_pred= mlp_tuned.predict(x_test)
dogruluk= accuracy_score(y_test,y_pred)
print(dogruluk)



