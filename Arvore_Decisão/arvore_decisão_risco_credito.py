# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 08:47:42 2021

@author: Walvi
"""
import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
print(classificador.feature_importances_)

export_graphviz(classificador,
                out_file= 'arvore.dot',
                feature_names=['historico', 'divida', 'garantias', 'renda'],
                class_names=['alto', 'moderado', 'baixo'],
                filled=True,
                leaves_parallel=True)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(resultado)
