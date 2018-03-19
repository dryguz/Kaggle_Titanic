# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:40:59 2018

@author: michal.drygajlo
"""
import sys, os
import numpy as np
import time
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
models = {'svm': [], 'log_reg': [], 'rand_forest': [], 'xgboost': []}

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
start = time.time()
rf_acc, rf_model = modeling.rand_forest_(X,y)
duration = time.time() - start

models['rand_forest_'].append(np.mean(rf_acc))
models['rand_forest_'].append(duration)
models['rand_forest_'].append(rf_model)

print('Score for Random Forest is {:.3}'.format(models['rand_forest_'][0]))



