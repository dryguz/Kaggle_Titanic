# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:51:50 2018

@author: michal.drygajlo

prepare for modeling
"""

def cleaning_data(df):
    '''
    check data types, fills NaN in Cabin column
    '''
    #Variables that are categorical - can be transform to dummy variable e.g.
    category_vars =['Pclass','Sex', '']
    print(category_vars)
    
    
    
    #Ticket variable needs cleaning:
    #- different types of Ticket number may regard different class 
    #- cross validate with Pclass
    df.Ticket
    
    
    
    #Cabin variable has a lot of NaN 
    #- which means many passanger didn't have a cabin at all 
    #- to be coded as  value e.g. 'no_cabin'
    #- cabin numbers have some code in them, 
    #- letters and numbers may mean different levels of ship...
    
    
    return df


def feature_engin(df):
    '''
    creating new features: cabin_letter, cabin_number
    '''
    
    
    return df