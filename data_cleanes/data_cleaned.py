import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
#  chargement des donnees 
df=pd.read_csv('./data/billets.csv',sep=";")
# correction des valeurs manquantes
df['margin_low'].fillna(value=df['margin_low'].mean(),inplace=True)
df.to_csv('./data_cleanes/data_cleaned.csv', index=False)
