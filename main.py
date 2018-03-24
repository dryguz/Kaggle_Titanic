# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:40:59 2018

@author: michal.drygajlo
"""
import sys, os
import numpy as np
import time
from sklearn import svm

path = '/home/mdrygajlo/PycharmProjects/Kaggle_Titanic/'
sys.path.append(path)
os.chdir(path)

from download_data import import_data
from prepare_data import prepare_data
import modeling

cleaner = prepare_data()
# download data
train, test = import_data()

# prepare data
X, y = cleaner.fit(train)

# set dictionary with models
models = {'svm': [],
          'svm_class': [],
          'log_reg': [], 
          'rand_forest': [], 
          'xgboost': []}

# first one - svm
start = time.time()
svm_acc, svm_model = modeling.svm_(X, y)
duration = time.time() - start

models['svm'].append(np.mean(svm_acc))
models['svm'].append(duration)
models['svm'].append(svm_model)

print('Score for SVM model is {:.3}'.format(models['svm'][0]))

# second one - logical regression
start = time.time()
lg_acc, lg_model = modeling.log_reg_(X,y)
duration = time.time() - start

models['log_reg'].append(np.mean(lg_acc))
models['log_reg'].append(duration)
models['log_reg'].append(lg_model)

print('Score for Logical Regression is {:.3}'.format(models['log_reg'][0]))

#third one - random forest
#start = time.time()
#rf_acc, rf_model = modeling.rand_forest_(X,y)
#duration = time.time() - start

#models['rand_forest_'].append(np.mean(rf_acc))
#models['rand_forest_'].append(duration)
#models['rand_forest_'].append(rf_model)

#print('Score for Random Forest is {:.3}'.format(models['rand_forest_'][0]))

# ----------------------------------------------------------------------------
# test svm with class

svm_parameters = {'Estimator__C': np.linspace(1, 1000, num=50),
              'Estimator__gamma': np.linspace(0.0001, 0.1, num=50)}

svm_model = svm.SVC(kernel = 'rbf',
                  random_state=123)
        
_svm_ = modeling.model_(svm_model, svm_parameters)

start = time.time()

_svm_model = _svm_.fit(X, y)
_svm_acc = _svm_.score(X, y)

duration = time.time() - start

models['svm_class'].append(np.mean(_svm_acc))
models['svm_class'].append(duration)
models['svm_class'].append(_svm_model)

print('Score for SVM model is {:.3}'.format(models['svm_class'][0]))