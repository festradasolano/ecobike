import math

from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Constantes
APP_PATH = "/git/ecobike/python/"
MODEL_NAME = "app_model.pkl"
SCALER_NAME = "app_scaler.pkl"

# Valores medios de altitud para reemplazar valores perdidos
MEAN_ALT_MEAN = 237.91719764827465
MEAN_ALT_STD = 95.75080110955669
MEAN_ALT_MEDIAN = 222.62181082400872
MEAN_ALT_MIN = 108.93285236256986
MEAN_ALT_MAX = 621.533255240702
MEAN_ALT_Q1 = 177.46499236113848
MEAN_ALT_Q2 = 222.62181082400872
MEAN_ALT_Q3 = 279.8545871631099
MEAN_ALT_SPEED_MEAN = 0.8877477575007959
MEAN_ALT_SPEED_STD = 2.8300428553999724
MEAN_ALT_SPEED_MEDIAN = 0.34198552880676647
MEAN_ALT_SPEED_MIN = 0.003306266838421906
MEAN_ALT_SPEED_MAX = 63.35427196937614
MEAN_ALT_SPEED_Q1 = 0.08954098936221333
MEAN_ALT_SPEED_Q2 = 0.34198552880676647
MEAN_ALT_SPEED_Q3 = 0.9173111218901985

# Aplicación Web
app = Flask(__name__)

# Cargar modelo de clasificación y escalador
home_path = str(Path.home())

clf = joblib.load(home_path + APP_PATH + MODEL_NAME)
app.logger.info("Modelo de clasificación '%s' cargado", MODEL_NAME)

scaler = joblib.load(home_path + APP_PATH + SCALER_NAME)
app.logger.info("Escalador '%s' cargado", SCALER_NAME)

# Ruta de prueba
@app.route('/')
def hello():
    return jsonify([{"response": "Hello, EcoBike!"}])

# API para analizar datos CSV del recorrido
@app.post("/usuarios/<string:user_id>/recorridos/<string:record_id>")
def analyze_csvdata(user_id, record_id):
    try:
        app.logger.info("Analizando recorrido '%s' del usuario '%s'", record_id, user_id)

        # Leer datos CSV del recorrido en la solicitud
        request_data = request.get_json()
        csv_string = request_data["data"]

        # Construir dataframe a partir de los datos CSV del recorrido
        csv_stringIO = StringIO(csv_string)
        df_csv = pd.read_csv(csv_stringIO, sep=",")

        # Generar atributos de la instancia del recorrido a analizar
        features = get_features(df_csv)
        df_features = pd.DataFrame([features])

        # Reemplazar valores perdidos en los atributos de altitud
        df_features = replace_lost_altitude(df_features)

        # Escalar atributos de instancia
        x = scaler.transform(df_features.iloc[:,:])

        # Clasificar instancia de recorrido
        y_pred = clf.predict(x)

        # Retornar clasificación
        app.logger.info("Recorrido '%s' del usuario '%s' ha sido clasificado como '%s'", record_id, user_id, y_pred[0])
        return jsonify([{"clase": y_pred[0]}])
    
    except:
        return jsonify([{"clase": "error"}]), 500


# Genera los atributos de una instancia a partir de un dataframe del recorrido
def get_features(df_csv):
    # Inicializar variables y listas
    total_distance, total_time = 0, 0
    speeds, altitudes, alt_speeds = list(), list(), list()

    # Recorrer dataframe
    for index, row in df_csv.iterrows():
        # Leer datos de inicio
        if index == 0:
            # Posición y tiempo
            x0 = (row["latitud"], row["longitud"])
            t0 = row["tiempo"]

            # Altura
            alt0 = row["altitud"]
            alt_t0 = t0
            if alt0 >= 0:
                altitudes.append(alt0)
            continue

        # Procesar distancia, tiempo y velocidad
        x = (row["latitud"], row["longitud"])
        t = row["tiempo"]
        distance = abs(haversine(x0[0], x0[1], x[0], x[1]))    # distancia en metros
        time = (t - t0) / 1000    # tiempo en segundos
        if time > 0:
            speed = distance / time    # velocidad en metros/segundo
            speeds.append(speed)
            total_distance += distance
            total_time += time
            x0 = x
            t0 = t

        # Procesar altitud, teniendo en cuenta valores no válidos
        alt = row["altitud"]
        if alt >= 0:
            altitudes.append(alt)
            if alt0 >= 0:
                delta_alt = abs(alt - alt0)    # cambio de altura en pies
                alt_time = (t - alt_t0) / 1000    # tiempo en segundos
                if alt_time > 0:
                    alt_speed = delta_alt / alt_time    # velocidad altura en pies/segundo
                    alt_speeds.append(alt_speed)
            alt0 = alt
            alt_t0 = t

    # Construir características de la instancia
    features = {}

    # Distancia y tiempo total
    features["distance"] = total_distance
    features["time"] = round(total_time)

    # Estadísticas de velocidad
    speed_mean, speed_std, speed_median, speed_min, speed_max, speed_q1, speed_q2, speed_q3 = get_statistics(speeds)
    features["speed_mean"] = speed_mean
    features["speed_std"] = speed_std
    features["speed_median"] = speed_median
    features["speed_min"] = speed_min
    features["speed_max"] = speed_max
    features["speed_q1"] = speed_q1
    features["speed_q2"] = speed_q2
    features["speed_q3"] = speed_q3
    features["speed_count"] = len(speeds)

    # Estadísticas de altura
    alt_mean, alt_std, alt_median, alt_min, alt_max, alt_q1, alt_q2, alt_q3 = get_statistics(altitudes)
    features["alt_mean"] = alt_mean
    features["alt_std"] = alt_std
    features["alt_median"] = alt_median
    features["alt_min"] = alt_min
    features["alt_max"] = alt_max
    features["alt_q1"] = alt_q1
    features["alt_q2"] = alt_q2
    features["alt_q3"] = alt_q3
    features["alt_count"] = len(altitudes)

    # Estadísticas de velocidad de altura
    alt_speed_mean, alt_speed_std, alt_speed_median, alt_speed_min, alt_speed_max, alt_speed_q1, alt_speed_q2, alt_speed_q3 = get_statistics(alt_speeds)
    features["alt_speed_mean"] = alt_speed_mean
    features["alt_speed_std"] = alt_speed_std
    features["alt_speed_median"] = alt_speed_median
    features["alt_speed_min"] = alt_speed_min
    features["alt_speed_max"] = alt_speed_max
    features["alt_speed_q1"] = alt_speed_q1
    features["alt_speed_q2"] = alt_speed_q2
    features["alt_speed_q3"] = alt_speed_q3
    features["alt_speed_count"] = len(alt_speeds)

    return features


# Distancia entre dos puntos, utilizando la latitud y longitud de cada punto
def haversine(lat1, lon1, lat2, lon2):
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598

    # Fórmula Haresine
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia_metros=(2*R*math.asin(math.sqrt(a)))*1000

    return distancia_metros


# Genera las estadísticas descriptivas de una lista de datos
def get_statistics(data_list):
    if len(data_list) == 0:
        return -1, -1, -1, -1, -1, -1, -1, -1
    data_array = np.array(data_list)
    mean = np.mean(data_array)
    std = np.std(data_array)
    median = np.median(data_array)
    min = np.min(data_array)
    max = np.max(data_array)
    q1 = np.percentile(data_array, 25)
    q2 = np.percentile(data_array, 50)
    q3 = np.percentile(data_array, 75)
    return mean, std, median, min, max, q1, q2, q3


# Reemplaza los valores perdidos de altitud en los atributos de la instancia
def replace_lost_altitude(df_features):
    # Reemplazar -1.0 por NaN
    df_features.replace(-1.0, np.nan, inplace = True)

    # Reemplazar NaN por valores medios
    df_features['alt_mean'].fillna(MEAN_ALT_MEAN, inplace=True)
    df_features['alt_std'].fillna(MEAN_ALT_STD, inplace=True)
    df_features['alt_median'].fillna(MEAN_ALT_MEDIAN, inplace=True)
    df_features['alt_min'].fillna(MEAN_ALT_MIN, inplace=True)
    df_features['alt_max'].fillna(MEAN_ALT_MAX, inplace=True)
    df_features['alt_q1'].fillna(MEAN_ALT_Q1, inplace=True)
    df_features['alt_q2'].fillna(MEAN_ALT_Q2, inplace=True)
    df_features['alt_q3'].fillna(MEAN_ALT_Q3, inplace=True)
    df_features['alt_speed_mean'].fillna(MEAN_ALT_SPEED_MEAN, inplace=True)
    df_features['alt_speed_std'].fillna(MEAN_ALT_SPEED_STD, inplace=True)
    df_features['alt_speed_median'].fillna(MEAN_ALT_SPEED_MEDIAN, inplace=True)
    df_features['alt_speed_min'].fillna(MEAN_ALT_SPEED_MIN, inplace=True)
    df_features['alt_speed_max'].fillna(MEAN_ALT_SPEED_MAX, inplace=True)
    df_features['alt_speed_q1'].fillna(MEAN_ALT_SPEED_Q1, inplace=True)
    df_features['alt_speed_q2'].fillna(MEAN_ALT_SPEED_Q2, inplace=True)
    df_features['alt_speed_q3'].fillna(MEAN_ALT_SPEED_Q3, inplace=True)

    return df_features