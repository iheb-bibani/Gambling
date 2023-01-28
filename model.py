import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import warnings
import joblib
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')

df = pd.read_excel('colors.xlsx')

df = df.sort_index(ascending=False)

df['Result'] = df['Result'].replace({'Green':0,'Red':1,'Green, Violet':2,'Violet':3,'Red, Violet':4})

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
 n_vars = 1 if type(data) is list else data.shape[1]
 df = pd.DataFrame(data)
 cols, names = list(), list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
     cols.append(df.shift(i))
     names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
 # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
     cols.append(df.shift(-i))
 if i == 0:
     names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
 else:
     names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
 # put it all together
 agg = pd.concat(cols, axis=1)
 agg.columns = names
 # drop rows with NaN values
 if dropnan:
     agg.dropna(inplace=True)
 return agg

df = df.drop(['Period','Price','Number'],axis=1)

X = series_to_supervised(df.shift(), n_in=26, n_out=1, dropnan=False)
y = df['Result']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

X_train = X_train.fillna(X_train.mode().iloc[0])
X_test = X_test.fillna(X_test.mode().iloc[0])

parameters = {'n_estimators':np.arange(1,100,1),
             'max_samples':np.arange(1,100,1),

             'bootstrap': [True,False],
             'bootstrap_features':[True,False],
             }
rf = BalancedBaggingClassifier()
rf = RandomizedSearchCV(rf, parameters)

rf.fit(X_train,y_train)

predictions = rf.predict(X_test)

filename = 'finalized_model.sav'
joblib.dump(rf, filename)

loaded_model = joblib.load(filename)
