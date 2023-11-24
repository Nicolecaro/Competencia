# **Nicole Stefanie Caro Rodriguez (63211190)**
# **Daniel Alfonso Lopez Sierra (64191108)**
"""

!pip install wooldridge

!pip install catboost

!wget -q -nc

import wooldridge as wd
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from xgboost import XGBRegressor

df=pd.read_csv('/content/data_wage_train.csv')
df.head()

y_train = df['wage']
x_train = df.drop(columns='wage')

linreg= LinearRegression()
linreg.fit(x_train, y_train)

x_test = pd.read_csv('/content/data_test.csv')

linreg_pred = linreg.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':linreg_pred}).to_csv('sol.csv', index=False)

ridge_reg= Ridge()
param_grid = {'alpha' : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}
g_ridge = GridSearchCV(
    ridge_reg,
    param_grid=param_grid,
    scoring= 'neg_mean_squared_error',
    cv=2, verbose=8
)

g_ridge.fit(x_train, y_train)

ridge_p= g_ridge.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':ridge_p}).to_csv('ridge1.csv', index=False)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error


!pip install wooldridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import wooldridge as wd

y = df['wage']
x = df[['educ',	'exper',	'married',	'female',	'tenure']]

tree = DecisionTreeRegressor()
tree.fit(x, y)
y_pred = tree.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':y_pred}).to_csv('tree.csv', index=False)

param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8],
              'min_samples_split': [4, 5, 6, 7, 8]}

tree_grid = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=2)

tree_grid.fit(x, y)

y_pred = tree_grid.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':y_pred}).to_csv('treem.csv', index=False)

rf = RandomForestRegressor()
rf.fit(x, y)
pred = rf.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':pred}).to_csv('randomf.csv', index=False)

param_grid = {'n_estimators':[50, 100, 150, 200],
              'max_depth': [4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [5, 10, 15, 20],
              'min_samples_leaf': [5, 10, 15, 20]}

rf_grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
rf_grid.fit(x, y)
pred = rf_grid.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':pred}).to_csv('randomf2.csv', index=False)

param_grid ={"max_depth": [5,6,7],
             "eta": [0.001, 0.01, 0.1],
             "subsample": [ 0.8, 0.9, 1],
             "colsample_bytree": [0.7, 0.6, 0.5],
             "n_estimators": [ 90, 150, 200]}

xgb_grid =GridSearchCV (XGBRegressor(), scoring= "neg_mean_squared_error", param_grid=param_grid, cv=2, verbose=1)
xgb_grid.fit (x,y)
preds = xgb_grid.predict (x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':pred}).to_csv('rf2.csv', index=False)

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
hgbc = HistGradientBoostingRegressor()
hgbc.fit(x,y)
y_pred_hgbc = hgbc.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':y_pred_hgbc}).to_csv('hist.csv', index=False)

param_grid = {'max_depht': [3,4,5],
              'eta': [0.01],
              'subsample': [0.8,0.9,1],
              'colsample_bytree': [0.8,0.7,0.6],
              'n_estimators': [200, 300]}
xgb_grid = GridSearchCV(XGBRegressor(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv=5, verbose=2)
xgb_grid.fit(x,y)
pred=xgb_grid.predict(x_test)

xgb_grid.best_params_

pd.DataFrame({'id': range(0,106),
              'Expected':pred}).to_csv('xbc6.csv', index=False)

from lightgbm.sklearn import LGBMRegressor
lgbc = LGBMRegressor()
lgbc.fit(x,y)
lgbc = lgbc.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':lgbc}).to_csv('lgbc.csv', index=False)

param_grid = {'max_depht': [3,4,5],
              'eta': [0.01],
              'subsample': [0.8,0.9,1],
              'colsample_bytree': [0.8,0.7,0.6],
              'n_estimators': [200, 300]}
lgbm_grid = GridSearchCV(LGBMRegressor(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv=5, verbose=2)
lgbm_grid.fit(x,y)
predlgbm=xgb_grid.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':predlgbm}).to_csv('L2.csv', index=False)

"""#**MODELO CON LA MEJOR PUNTUACIÓN**"""

param_grid ={"max_depth": [5,6,7],
             "eta": [0.001, 0.01, 0.1],
             "subsample": [ 0.8, 0.9, 1],
             "colsample_bytree": [0.7, 0.6, 0.5],
             "n_estimators": [ 90, 150, 200]}

lgbm_grid = GridSearchCV(LGBMRegressor(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv=2, verbose=3)
lgbm_grid.fit(x,y)
predlgbm=xgb_grid.predict(x_test)

pd.DataFrame({'id': range(0,106),
              'Expected':predlgbm}).to_csv('lgbm3.csv', index=False)
