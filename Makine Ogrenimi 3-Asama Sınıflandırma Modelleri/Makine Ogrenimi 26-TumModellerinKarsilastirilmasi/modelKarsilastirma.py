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
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier


df= pd.read_csv("diabetes.csv")
print(df.head())
y= df["Outcome"]
x= df.drop(['Outcome'], axis=1)
x_train, x_test, y_train, y_test= train_test_split(x,
                                                   y,
                                                   test_size=0.30,
                                                   random_state=42)

#Model
#%%
loj= LogisticRegression(solver="liblinear")
loj_model= loj.fit(x_train,y_train)

knn= KNeighborsClassifier(n_neighbors=11)
knn_tuned= knn.fit(x_train,y_train)

svm= SVC(C=2,kernel="linear")
svm_tuned= svm.fit(x_train,y_train)

mlp= MLPClassifier(solver="lbfgs",activation="logistic",alpha=1,hidden_layer_sizes=(3,5))
mlpc_tuned= mlp.fit(x_train,y_train)

dtc= DecisionTreeClassifier(max_depth=5,min_samples_split=20) 
cart_tuned= dtc.fit(x_train,y_train)

rf= RandomForestClassifier(max_features=7, min_samples_split=5,n_estimators=500)
rf_tuned= rf.fit(x_train,y_train)

gbm= GradientBoostingClassifier(learning_rate=0.01,max_depth=5,n_estimators=500)
gbm_tuned= gbm.fit(x_train,y_train)

xgb= XGBClassifier(learning_rate=0.001,max_depth=5,n_estimators=2000,subsample=1)
xgb_tuned= xgb.fit(x_train,y_train)

lgbm= LGBMClassifier(learning_rate=0.01,max_depth=1,n_estimators=500)
lgbm_tuned= lgbm.fit(x_train,y_train)

catb= CatBoostClassifier(depth=8,iterations=200,learning_rate=0.01)
catb_tuned= catb.fit(x_train,y_train,verbose=False)

modeller= [
    knn_tuned,
    loj_model,
    svm_tuned,
    mlpc_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    catb_tuned,
    lgbm_tuned,
    xgb_tuned]

sonuc= []
sonuclar= pd.DataFrame(columns= ["Modüller","Accuracy"])

for model in modeller:
    isimler= model.__class__.__name__
    y_pred= model.predict(x_test)
    dogruluk= accuracy_score(y_test,y_pred)
    sonuc= pd.DataFrame([[isimler,dogruluk*100]],columns= ["Modeller","Accuracy"])
    sonuclar= sonuclar.append(sonuc)

print(sonuclar)
sns.barplot(x='Accuracy', y='Modeller', data=sonuclar, color='r')
plt.xlabel('Accuracy%')
plt.title('Modellerin Doğruluk Oranları')
