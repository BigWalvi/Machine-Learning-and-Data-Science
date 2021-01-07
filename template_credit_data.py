# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:22:29 2021

@author: Walvi
"""
import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Dividindo dados de Treinamento e de Teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from collections import Counter
Counter(classe_teste)


























