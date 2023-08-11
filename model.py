
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
df = pd.read_csv('./data_cleanes/data_cleaned.csv', sep=",")
#df['is_genuine'].astype('int')
from sklearn.model_selection import train_test_split
x=df[["diagonal","height_left","height_right","margin_low","margin_up","length"]]
y=df["is_genuine"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#entrainement du model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
# Prédictions sur l'ensemble de test
y_pred = model.predict(x_test)
#evaluation du modele
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# Affichage des résultats
print("Précision du modèle : {:.2f}%".format(accuracy * 100))
print("\nMatrice de confusion :")
print(conf_matrix)
print("\nRapport de classification :")
print(class_report)

