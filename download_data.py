# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:08:19 2018

@author: michal.drygajlo

Module saving data to drive
"""




def import_data():
    
    import pandas as pd
    
    data_path = 'data/'
    train_file = 'train.csv'
    test_file = 'test.csv'
    
    train = pd.read_csv(data_path + train_file)
    test = pd.read_csv(data_path + test_file)
    
    return train, test
