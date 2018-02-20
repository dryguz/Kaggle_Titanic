# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:48:34 2018

@author: michal.drygajlo

['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Cabin_deck']
"""

import download_data
train, test = download_data.import_data()

import prepare_data
df = prepare_data.cleaning_data(train)

import seaborn as sns
import matplotlib.pyplot as plt

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

sns.set_context("paper")

_ = plt.plot()

sns.swarmplot(x='Sex', y='Fare', hue='Survived', data=df)

sns.violinplot(x='Cabin_deck', y='Fare', hue='Sex', data=df, split=False,
               inner="quart", palette={"male": "b", "female": "y"})

sns.jointplot(df.Cabin_deck, df.Pclass, kind='kde', size=7, space=0)

sns.color_palette("cubehelix", 8)


sns.pointplot(x='Pclass',y='Survived', hue='Sex', data=df, )


plt.plot()
