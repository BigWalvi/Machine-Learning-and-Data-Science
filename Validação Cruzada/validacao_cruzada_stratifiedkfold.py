# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:45:58 2021

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

from sklearn.naive_bayes import GaussianNB

import numpy as np
a = np.zeros(5)
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0],1))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
kfold = StratifiedKFold(n_splits=10, shuffle=(True), random_state=(0))
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0],1))):
    #print(f'Índice treinamento: {indice_treinamento} Índice teste: {indice_teste}')
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

matriz_final = np.mean(matrizes, axis=0)
resultados = np.asarray(resultados)
resultados.mean()
resultados.std()















