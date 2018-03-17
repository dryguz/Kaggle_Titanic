# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:51:50 2018

@author: michal.drygajlo

prepare for modeling
"""
import pandas as pd


def prepare_set(df):

    """
    check data types, fills NaN in Cabin column
    """
    
    # drop columns
    for each in ['PassengerId' 'Name']:
        try:
            df.drop(each, axis=1, inplace=True)
        except ValueError:
            pass

    # casting type of data
    df.Pclass = df.Pclass.astype('category')
    df.Sex = df.Sex.astype('category')
    df.Survived = df.Survived.astype('category')

    # Cabin variable has a lot of NaN
    # - which means many passenger didn't have a cabin at all
    # - to be coded as  value e.g. '00'
    df.Cabin.fillna('00', inplace=True)
    
    # - cabin numbers have some code in them,
    # - after analysis - first letter is the signiture of a titanic's deck
    # signatures of the decks are from A to G
    # passengers without cabin are assigned letter Z
    cabins = df.Cabin.str.split(' ', expand=True)
    
    df['Cabin_deck'] = 'Z'
    for col in cabins.columns:
        deck = cabins.loc[:,col].str.extract('([A-Z])',expand=False)
        indx = df['Cabin_deck'] > deck
        df.loc[indx,'Cabin_deck'] = deck[indx]
    
    df.Cabin_deck = df.Cabin_deck.astype('category')        
    
    # age nan fill with a mean from group sex/cabin_deck
    male_indx = df.Sex == 'male'
    cabin_z_indx = df.Cabin_deck == 'Z'
    
    indx1 = male_indx & cabin_z_indx
    m1 = int(df.loc[indx1,'Age'].mean(skipna=True))
    df.loc[indx1,'Age'] = df.loc[indx1,'Age'].fillna(m1)
    
    indx2 = ~male_indx & cabin_z_indx
    m2 = int(df.loc[indx2,'Age'].mean(skipna=True))
    df.loc[indx2,'Age'] = df.loc[indx2,'Age'].fillna(m2)
    
    indx3 = ~male_indx & ~cabin_z_indx
    m3 = int(df.loc[indx3,'Age'].mean(skipna=True))
    df.loc[indx3,'Age'] = df.loc[indx3,'Age'].fillna(m3)
    
    indx4 = male_indx & ~cabin_z_indx
    m4 = int(df.loc[indx4,'Age'].mean(skipna=True))
    df.loc[indx4,'Age'] = df.loc[indx4,'Age'].fillna(m4)

    return df
