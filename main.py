# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:40:59 2018

@author: michal.drygajlo
"""

import download_data
train, test = download_data.import_data()


import prepare_data

X = prepare_data.cleaning_data(train)
X = prepare_data.feature_engin(X)

import modeling

model_types = ['svn', 'log_reg', 'rand_forest']
accuracies = []
models = []

modeling.svn_model(X)
