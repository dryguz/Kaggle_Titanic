# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:58:02 2018

@author: michal.drygajlo

create classifications model 
"""
# ---------------------------------------------------------------------------
# svn_model

def svm_(df):
'''
training support vector machines for classification
kernel is rbf, optimal C is search for in [1, 10, 100, 1000] 

'''
    from sklearn.preprocessing import Imputer, StandardScaler    
    from sklearn.pipeline import Pipeline    
    from sklearn import svm
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    
    clf = svm.SVC(kernel='rbf')
    
    steps = [('imputation',Imputer(strategy='NaN', axis=0)),
             ('scaler', StandardScaler())
             ('dummy',dummy),
             ('clf':clf)]
    

    svm_cls = Pipeline(steps)
    
    parameters = {'clf__C':np.linespace(1,1000,num=50),
                  'clf__gamma':np.linespace(0.0001,0.1,num=50)}
    
    rm_clf = RandomizedSearchCV(pipeline, param_distribution=parameters)
    rm_clf
    
    return svm_acc, svm_model


# ---------------------------------------------------------------------------
# log_reg_model
    
def log_reg_(df):
    
    return log_reg_acc, log_reg_model


# ---------------------------------------------------------------------------
# rand_forest_model
    
def rand_forest_(df):
    
    return rand_forest_model, rand_forest_model

# ---------------------------------------------------------------------------
# xgboost_model
    
def xgboost_(df):
    
    return xgb_acc, xgb_model