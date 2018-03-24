# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:40:59 2018

@author: michal.drygajlo
"""
import sys, os
import numpy as np
import time
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier

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
          'log_reg': [], 
          'rand_forest': [], 
          'xgboost': []}

#-----------------------------------------------------------------------------
# first one - svm

svm_parameters = {'Estimator__C': np.linspace(1, 1000, num=50),
              'Estimator__gamma': np.linspace(0.0001, 0.1, num=50)}

svm_ = svm.SVC(kernel = 'rbf',
                  random_state=123)
        
svm_ = modeling.model_(svm_, svm_parameters)

start = time.time()

svm_model = svm_.fit(X, y)
svm_acc = svm_.score(X, y)

duration = time.time() - start

models['svm'].append(np.mean(svm_acc))
models['svm'].append(duration)
models['svm'].append(svm_model)

print('Score for SVM model is {:.3}'.format(models['svm'][0]))

#-----------------------------------------------------------------------------
# second one - logical regression

sgd_parameters = {'Estimator__alpha': np.linspace(0.0001, 0.1, 50),
                  'Estimator__l1_ratio': np.linspace(0, 1, 50)
                  }

sgd_ = linear_model.SGDClassifier(loss='log', 
                                     penalty='elasticnet',
                                     tol=0.001,
                                     random_state=123)

sgd_ = modeling.model_(sgd_, sgd_parameters)

start = time.time()

sgd_model = sgd_.fit(X, y)
sgd_acc = sgd_.score(X, y)

duration = time.time() - start

models['log_reg'].append(np.mean(sgd_acc))
models['log_reg'].append(duration)
models['log_reg'].append(sgd_model)

print('Score for Logical Regression is {:.3}'.format(models['log_reg'][0]))

#-----------------------------------------------------------------------------
#third one - random forest

rf_parameters = {'Estimator__n_estimators': np.linspace(5, 400, 30, dtype='int'),
                 'Estimator__max_features': np.linspace(0.02, 1.0, num=30)#,
#                 'Estimator__min_samples_split': np.linspace(2, 10, 1),
#                 'Estimator__min_samples_leaf': np.linspace(1, 10, 1)
                 }

rf_model = RandomForestClassifier()

rf_model = modeling.model_(rf_model, rf_parameters) 

start = time.time()

rf_model = rf_model.fit(X, y)
rf_acc = rf_model.score(X, y)

duration = time.time() - start

models['rand_forest'].append(np.mean(rf_acc))
models['rand_forest'].append(duration)
models['rand_forest'].append(rf_model)

print('Score for Random Forest is {:.3}'.format(models['rand_forest'][0]))




