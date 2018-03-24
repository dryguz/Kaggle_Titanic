# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:58:02 2018

@author: michal.drygajlo

create classifications model 
"""

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion 
import pandas as pd

# ---------------------------------------------------------------------------
# class for models, with training method, predict method


class model_:
    """
    Class model_ takes as parameter a scikit-learn API model 
    and dictonery with parameters for RandomSearch. 
    Because of using RS dictonary must have keys following pattern: 
    {'Estimator__parameter': range_of_values}
    
    Class has two methods: fit(X,y) and score(X,y)
    -fit() returns RandomSearch class model
    -score() return 5 accuracy values from 5 times fold cross-validation
    """
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
        