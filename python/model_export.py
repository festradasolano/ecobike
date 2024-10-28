import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

# CONSTANTES
DF_PATH = "datasets/df_robust.csv"
RANDOM_STATE = 1
NOBIKE_LABEL = "no_bike"
MODEL_NAME = "ecobike_model.pkl"

# Leer datos CSV
df = pd.read_csv(DF_PATH, index_col=0)

# Unir clases diferentes a 'bike'
df.loc[df["label"] != "bike", "label"] = NOBIKE_LABEL

print("Conjunto de datos cargado!")

# Separar características y clase
x = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1:].to_numpy().ravel()

# Balancear el número de instancias de la clase minoritaria 'bike'
sm = SMOTE(random_state=RANDOM_STATE)
x_res, y_res = sm.fit_resample(x, y)

# Entrenar el modelo con el conjunto de datos balanceado
clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
clf.fit(x_res, y_res)

print("Modelo de clasificación binaria entrenado!")

# Exportar el modelo a un archivo pickle
joblib.dump(clf, MODEL_NAME)

print("Modelo de clasificación binaria exportado!")
