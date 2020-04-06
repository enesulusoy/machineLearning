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


#Model ve Tahmin
#%%
print(df["Outcome"].value_counts()) #bağımlı değişken oranlarını gösterir
print(df.describe().T)


y= df["Outcome"]
x= df.drop(["Outcome"],axis=1)
print(y.head())
print(x.head())


loj= LogisticRegression(solver="liblinear")
loj_model= loj.fit(x,y)
print(loj_model.intercept_) #katsayı gösterir
print(loj_model.coef_)  #bağımsız değişken katsayıları

y_pred= loj_model.predict(x)
print(y_pred)
print(confusion_matrix(y,y_pred))   #karmaşıklık matrisini gösterir
print(accuracy_score(y,y_pred)) #doğruluk oranını verir    
print(classification_report(y,y_pred))

#tahmin işlemlerini 1 ve 0 dan değilde olasılık değerlerinden almak istersek
print(loj_model.predict_proba(x)[0:10])

logit_roc_auc= roc_auc_score(y,loj_model.predict(x))    #gerçek değerler ile tahmin edilen değerler arasında roc score oluşturduk
fpr, tpr, thresholds= roc_curve(y, loj_model.predict_proba(x)[:,1]) #eğri oluşturmak için olasılık ve gerçek değerleri üzerinden false pozitif rate ve true pozitif rate belirlenmesi
plt.figure()    
plt.plot(fpr,tpr,label='AUC (area = %0.2f)' % logit_roc_auc)    #labelleri ifade eden kısım
plt.plot([0,1],[0,1],'r--') #eksen ayarlama
plt.xlim([0.0,1.0])         #eksen ayarlama
plt.ylim([0.0,1.05])        #eksen ayarlama
plt.xlabel('False Positive Rate')   #isimlendirme
plt.ylabel('True Positive Rate')    #isimlendirme
plt.title('Receiver operating characteristic')  #isimlendirme
plt.legend(loc="lower right")   #sağ alt bilgi kutucuğu
plt.savefig('Log_ROC')  #kayıt işlemi
plt.show()  #görseli gösterme


#Model Tuning (Model Doğrulama)
#%%
x_train, x_test, y_train, y_test= train_test_split(x,
                                                   y,
                                                   test_size=0.30,
                                                   random_state=42)
loj= LogisticRegression(solver="liblinear")
loj_model= loj.fit(x_train,y_train)
y_pred= loj_model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(cross_val_score(loj_model,x_test,y_test,cv=10))   #10 farklı doğruluk oranı
print(cross_val_score(loj_model,x_test,y_test,cv=10).mean())    #10 farklı doğruluk oranın ortalaması

