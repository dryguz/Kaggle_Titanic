# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:40:59 2018

@author: michal.drygajlo
"""

import download_data
import modeling
import prepare_data

# download data
train, test = download_data.import_data()

# prepare data
X = prepare_data.prepare_set(train)

# set dictionary with models
models = {'svm': [], 'log_reg': [], 'rand_forest': [], 'xgboost': []}

# first one - svm
svm_acc, svm_model = modeling.svm_(X)
models['svm'].append(svm_acc)
models['svm'].append(svm_model)

print(models['svm'][0])


