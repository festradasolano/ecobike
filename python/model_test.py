import joblib
import pandas as pd

# CONSTANTES
DF_NAME = "df_robust.csv"
NOBIKE_LABEL = "no_bike"
MODEL_NAME = "app_model.pkl"

# Leer datos CSV
df = pd.read_csv(DF_NAME, index_col=0)

# Unir clases diferentes a 'bike'
df.loc[df["label"] != "bike", "label"] = NOBIKE_LABEL

print("Conjunto de datos cargado!")

# Importar el modelo desde un archivo pickle
clf = joblib.load(MODEL_NAME)

print("Modelo de clasificación binaria importado!")

# Separar características y clase
x = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1:].to_numpy().ravel()

# Prueba no_bike
y_pred = clf.predict(x[0:1])
print("Clase actual:",  y[0], "; Predicción:", y_pred[0])

# Prueba bike
y_pred = clf.predict(x[204:205])
print("Clase actual:",  y[204], "; Predicción:", y_pred[0])