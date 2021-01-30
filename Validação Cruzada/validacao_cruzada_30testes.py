# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:00:15 2021

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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits=10, shuffle=(True), random_state=(i))
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros(shape=(previsores.shape[0],1))):
        classificador = GaussianNB()
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
resultados30 = np.asarray(resultados30)
for res in resultados30:
    print(str(res).replace('.', ','))
med = resultados30.mean()
print(f'Média: {med:.5f}'.replace('.', ','))












