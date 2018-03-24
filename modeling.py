# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:58:02 2018

@author: michal.drygajlo

create classifications model 
"""

from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion 
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# class for models, with training method, predict method


class model_:
    def __init__(self, model, parameters_for_model):
        self.model = model
        self.parameters = parameters_for_model    
        self.for_scalling =[]
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        

        
        
        def select_to_scal(data):
            return data[:,self.for_scalling]
        
        
        def select_not_to_scal(data):
            return data[:,self.not_for_scalling]
        
        
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
                ('Estimator', self.model)
                ]
        
        self.ppl = Pipeline(steps=steps)
        
        self.clf = RandomizedSearchCV(self.ppl, 
                        param_distributions=self.parameters, 
                        n_iter=30,
                        random_state=123,                                
                        cv=5)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        for col in ['Age', 'SibSp', 'Parch', 'Fare']:
            self.for_scalling.append(self.X.columns.get_loc(col))
            
        self.not_for_scalling = list(set(range(len(self.X.columns)))-set(self.for_scalling))
        
        self.trained_model = self.clf.fit(self.X,self.y)
        
        return self.trained_model
    
    def score(self, X, y):
        self.X = X
        self.y = y
        
        
        
        self.models_score = cross_val_score(
                self.trained_model.best_estimator_, 
                self.X, 
                self.y, cv=5)
        return self.models_score
 

    
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
    
    svm_cls = svm.SVC(kernel = 'rbf',
                      random_state=123)
    
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
            ('Estimator', svm_cls)
            ]
    
    ppl = Pipeline(steps=steps)
    
    parameters = {'Estimator__C': np.linspace(1, 1000, num=50),
                  'Estimator__gamma': np.linspace(0.0001, 0.1, num=50)}
    
    rm_clf = RandomizedSearchCV(ppl, 
                                param_distributions=parameters, 
                                n_iter=30,
                                random_state=123,                                
                                cv=5)

    svm_model = rm_clf.fit(X, y)
    svm_acc = cross_val_score(svm_model.best_estimator_, X, y, cv=5)
    
    
    return svm_acc, svm_model


# ---------------------------------------------------------------------------
# log_reg_model
    
def log_reg_(X, y):
    
    for_scalling =[]
    
    for col in ['Age', 'SibSp', 'Parch', 'Fare']:
        for_scalling.append(X.columns.get_loc(col))
        
    not_for_scalling = list(set(range(len(X.columns)))-set(for_scalling))
    
    
    def select_to_scal(data):
        return data[:,for_scalling]
    
    
    def select_not_to_scal(data):
        return data[:,not_for_scalling]
    
    
    sgd_cls = linear_model.SGDClassifier(loss='log', 
                                         penalty='elasticnet',
                                         tol=0.001,
                                         random_state=123)
    
    
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
        ('Estimator', sgd_cls)
        ]
    
    ppl = Pipeline(steps=steps)
    
    
    parameters = {'Estimator__alpha': np.linspace(0.0001, 0.1, 50),
                  'Estimator__l1_ratio': np.linspace(0, 1, 50)
                  }
    
    rm_clf = RandomizedSearchCV(ppl, 
                                param_distributions=parameters, 
                                n_iter=30,
                                random_state=123,
                                cv=5)
    
    
    log_reg_model = rm_clf.fit(X, y)
    log_reg_acc  =  cross_val_score(log_reg_model.best_estimator_, X, y, cv=5)
    
    return log_reg_acc, log_reg_model


# ---------------------------------------------------------------------------
# rand_forest_model
    
def rand_forest_(X, y):
    
    for_scalling =[]
    
    for col in ['Age', 'SibSp', 'Parch', 'Fare']:
        for_scalling.append(X.columns.get_loc(col))
        
    not_for_scalling = list(set(range(len(X.columns)))-set(for_scalling))
    
    
    def select_to_scal(data):
        return data[:,for_scalling]
    
    def select_not_to_scal(data):
        return data[:,not_for_scalling]
    
    
    rf_clf = RandomForestClassifier(random_state=123
                                    
                                    )

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
        ('Estimator', rf_clf)
        ]
    
    ppl = Pipeline(steps=steps)


    parameters = {'Estimator__n_estimators': np.linspace(3, 100, 3),
                  'Estimator__': '',
                  'Estimator__': ''
            }
    
    rf_clf = RandomizedSearchCV(ppl, 
                                param_distributions=parameters, 
                                n_iter=30,
                                random_state=123,
                                cv=5)
    
    rand_forest_model, rand_forest_model = 1, 1
    return rand_forest_model, rand_forest_model

# ---------------------------------------------------------------------------
# xgboost_model
    
def xgboost_(df):
    xgb_acc, xgb_model = 1, 1
    return xgb_acc, xgb_model