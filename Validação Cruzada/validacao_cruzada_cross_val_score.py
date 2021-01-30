# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:37:07 2021

@author: warve
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

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

resultados = cross_val_score(classificador, previsores, classe, cv=10)
resultados.mean()
resultados.std()
