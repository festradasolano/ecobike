# TODO utilizar dataset limpio para generar escalador y exportar

import joblib
import pandas as pd

from sklearn.preprocessing import RobustScaler

# CONSTANTES
DF_PATH = "datasets/df_clean.csv"
SCALER_NAME = "app_scaler.pkl"

# Leer datos CSV
df = pd.read_csv(DF_PATH, index_col=0)

print("Conjunto de datos cargado!")

# Entrenar el escalador con las características del conjunto de datos
scaler = RobustScaler().fit(df.iloc[:,:-1])

print("Escalador entrenado!")

# Exportar el escalador a un archivo pickle
joblib.dump(scaler, SCALER_NAME)

print("Escalador exportado!")