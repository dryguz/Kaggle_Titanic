# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:58:02 2018

@author: michal.drygajlo

create classifications model 
"""

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion 
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# svn_model

def svm_(X, y):
    """
    training support vector machines for classification
    kernel is rbf, optimal C is search for in [1, 10, 100, 1000]
    """
    for_scalling =[]
    
    for col in ['Age', 'SibSp', 'Parch', 'Fare']:
        for_scalling.append(X.columns.get_loc(col))
        
    not_for_scalling = list(set(range(len(X.columns)))-set(for_scalling))
    
    
    def select_to_scal(data):
        return data[:,for_scalling]
    
    
    def select_not_to_scal(data):
        return data[:,not_for_scalling]
    
    #cls = svm.SVC(kernel = 'rbf')
    
    steps = [('Union', FeatureUnion(transformer_list = [
                ('for_scaler', Pipeline(steps =[
                        ('select_for_scaler', FunctionTransformer(
                                select_to_scal)
                                ),
                        ('StandardScaler', StandardScaler()
                                )])
                ),
                ('not_scaler', FunctionTransformer(select_not_to_scal))
                ])
            ),
            ('Estimator', svm.SVC(kernel = 'rbf'))
            ]
    
    ppl = Pipeline(steps=steps)
    
    parameters = {'Estimator__C': np.linspace(1, 1000, num=50),
                  'Estimator__gamma': np.linspace(0.0001, 0.1, num=50)}
    
    rm_clf = RandomizedSearchCV(ppl, 
                                param_distributions=parameters, 
                                n_iter=20,
                                cv=5)

    svm_model = rm_clf.fit(X, y)
    svm_acc = cross_val_score(svm_model.best_estimator_, X, y, cv=5)
    
    
    return svm_acc, svm_model


# ---------------------------------------------------------------------------
# log_reg_model
    
def log_reg_(df):
    
    log_reg_acc, log_reg_model = 1, 1
    return log_reg_acc, log_reg_model


# ---------------------------------------------------------------------------
# rand_forest_model
    
def rand_forest_(df):
    
    rand_forest_model, rand_forest_model = 1, 1
    return rand_forest_model, rand_forest_model

# ---------------------------------------------------------------------------
# xgboost_model
    
def xgboost_(df):
    xgb_acc, xgb_model = 1, 1
    return xgb_acc, xgb_model