# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 09:49:14 2021

@author: Walvi
"""
import pandas as pd

base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]
#apagar coluna
#base.drop('age', 1, inplace=True)

#apagar somente os registros com problemas
#base.drop(base[base.age < 0].index, inplace=True)

# Preencher os valores manualmente com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92
