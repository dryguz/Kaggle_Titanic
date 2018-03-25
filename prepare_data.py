# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:51:50 2018

@author: michal.drygajlo

prepare for modeling
"""
import pandas as pd
class prepare_data:
    
    
    def __init__(self):
        self.df = pd.DataFrame()
        
        
    def fit(self, data, columns_to_drop, with_y=True):
        
        """
        changes data types, fills NaN
        """
        
        # drop columns
        self.df = data.copy()
        for each in columns_to_drop:
            try:
                self.df.drop(each, axis=1, inplace=True)
            except ValueError:
                pass
    
        
        # Fare NaN fill with Pclass cross mean
        indx1 = self.df.Pclass ==1
        pclass1 = self.df[indx1].Fare.mean()
        self.df.loc[indx1, 'Fare'] = self.df.loc[indx1, 'Fare'].fillna(pclass1)
        
        indx2 = self.df.Pclass ==2
        pclass2 = self.df[indx2].Fare.mean()
        self.df.loc[indx2, 'Fare'] = self.df.loc[indx2, 'Fare'].fillna(pclass2)
        
        indx3 = self.df.Pclass ==3
        pclass3 = self.df[indx3].Fare.mean()
        self.df.loc[indx3, 'Fare'] = self.df.loc[indx3, 'Fare'].fillna(pclass3)    
            
        
        # casting type of data
        self.df.Embarked = self.df.Embarked.astype('object')
        self.df.Pclass = self.df.Pclass.astype('object')
        self.df.Sex = self.df.Sex.astype('category')

    
        # Cabin variable has a lot of NaN
        self.df.Cabin.fillna('00', inplace=True)
        
        
        # Cabin numbers have some code in them,
        cabins = self.df.Cabin.str.split(' ', expand=True)
        
        self.df['Cabin_deck'] = 'Z'
        for col in cabins.columns:
            deck = cabins.loc[:,col].str.extract('([A-Z])',expand=False)
            indx = self.df['Cabin_deck'] > deck
            self.df.loc[indx,'Cabin_deck'] = deck[indx]
        
        self.df.Cabin_deck = self.df.Cabin_deck.astype('category')        
        self.df.drop('Cabin', axis=1, inplace=True)
        
        
        # Age NaN fill with a mean from group sex/cabin_deck
        male_indx = self.df.Sex == 'male'
        cabin_z_indx = self.df.Cabin_deck == 'Z'
        
        indx1 = male_indx & cabin_z_indx
        m1 = int(self.df.loc[indx1,'Age'].mean(skipna=True))
        self.df.loc[indx1,'Age'] = self.df.loc[indx1,'Age'].fillna(m1)
        
        indx2 = ~male_indx & cabin_z_indx
        m2 = int(self.df.loc[indx2,'Age'].mean(skipna=True))
        self.df.loc[indx2,'Age'] = self.df.loc[indx2,'Age'].fillna(m2)
        
        indx3 = ~male_indx & ~cabin_z_indx
        m3 = int(self.df.loc[indx3,'Age'].mean(skipna=True))
        self.df.loc[indx3,'Age'] = self.df.loc[indx3,'Age'].fillna(m3)
        
        indx4 = male_indx & ~cabin_z_indx
        m4 = int(self.df.loc[indx4,'Age'].mean(skipna=True))
        self.df.loc[indx4,'Age'] = self.df.loc[indx4,'Age'].fillna(m4)
        
        
        # Are we preparing data with y column? 
        if with_y: 
            self.y = self.df.Survived
            self.X = pd.get_dummies(self.df.drop('Survived', axis=1))
            self.columns = list(self.X.columns)
        else:
            self.X = pd.get_dummies(self.df)
            self.missing_columns = set(self.columns) - set(self.X.columns)
            
            if len(self.missing_columns) > 0:
                for col in self.missing_columns:
                    self.X[col] = 0
                
        

    def get_X(self):
        return self.X
    
    
    def get_y(self):
        return self.y
    
