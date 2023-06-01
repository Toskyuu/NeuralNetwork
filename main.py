import glob
import os
import openpyxl
import xlsxwriter

from glob import glob
import pandas as pd
import tensorflow as tf
import numpy as np
import re
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt



def draw_curves(history, key1='accuracy', ylim1=(0.8, 1.00),
                key2='loss', ylim2=(0.0, 1.0)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r--")
    plt.plot(history.history['val_' + key2], "g--")
    plt.ylabel(key2)
    plt.xlabel('Epoch')
    plt.ylim(ylim2)
    plt.legend(['train', 'test'], loc='best')

    plt.show()

def load_data(col_names: list, file_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if file_type == 'dynamic':
        all_files = glob("pomiary/F*/*.xlsx")
        files = [f for f in all_files if not (re.search("stat", f) or re.search("random", f))]
    else:
        files = glob("pomiary/F*/*_stat_*.xlsx")
    data = [pd.read_excel(file, header=0, usecols=col_names) for file in files]
    data = pd.concat(data, ignore_index=True)
    data = data.dropna()
    measurement = pd.concat([data[col_names[0]], data[col_names[1]]], axis=1)
    reference = pd.concat([data[col_names[2]], data[col_names[3]]], axis=1)
    print(f"{file_type.capitalize()} data has been successfully loaded.")
    return measurement, reference


# DO POPRAWY
def calculate_mean_squared_err(measurement: pd.DataFrame, reference: pd.DataFrame, choice=None):
    reference = reference.values.tolist()
    mse = []
    measurement = measurement.tolist() if choice == "result" else measurement.values.tolist()
    for i in range(len(measurement)):
        value = mean_squared_error([measurement[i][0], measurement[i][1]], [
            reference[i][0], reference[i][1]])
        mse.append(math.sqrt(value))
    return np.sort(mse)


if __name__ == '__main__':
    # WCZYTANIE DANYCH
    (measurement, reference) = load_data(col_names=['data__coordinates__x', 'data__coordinates__y',
                                                    'reference__x', 'reference__y'], file_type="stat")
    (dynamic_measurement, dynamic_reference) = load_data(col_names=['data__coordinates__x', 'data__coordinates__y',
                                                                    'reference__x', 'reference__y'],
                                                         file_type="dynamic")

    # ZMIANA WSZYSTKICH WARTOŚCI "NAN" NA 0
    measurement.fillna(0, inplace=True)
    reference.fillna(0, inplace=True)
    dynamic_measurement.fillna(0, inplace=True)
    dynamic_reference.fillna(0, inplace=True)

    measurement = (measurement.astype('float32')) / 10000
    reference = (reference.astype('float32')) / 10000
    dynamic_measurement = (dynamic_measurement.astype('float32')) / 10000
    dynamic_reference = (dynamic_reference.astype('float32')) / 10000

    # UTWORZENIE SIECI I WARSTW
    network = tf.keras.models.Sequential()
    network.add(tf.keras.layers.Dense(512, activation='relu'))
    network.add(tf.keras.layers.Dropout(0.2))
    network.add(tf.keras.layers.Dense(256, activation='relu'))
    network.add(tf.keras.layers.Dropout(0.2))
    network.add(tf.keras.layers.Dense(128, activation='relu'))
    network.add(tf.keras.layers.Dropout(0.2))
    network.add(tf.keras.layers.Dense(64, activation='relu'))
    network.add(tf.keras.layers.Dense(32, activation='relu'))
    network.add(tf.keras.layers.Dense(16, activation='relu'))
    network.add(tf.keras.layers.Dense(8, activation='relu'))
    network.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    # KOMPILACJA SIECI PRZY POMOCY OPTYMALIZATORA "ADAM"
    network.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy'])

    EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              patience=5,
                              verbose=1)

    # NAUCZANIE SIECI PRZY POMOCY DANYCH STATYCZNYCH
    history = network.fit(measurement, reference, epochs=15,
                verbose=1, batch_size=256,
                callbacks= [EarlyStop])

    network.evaluate(dynamic_measurement, dynamic_reference, batch_size=512)

    # WYRYSOWANIE WYKRESÓW POMOCNICZYCH
    draw_curves(history, key1='accuracy', ylim1=(0.9, 1.0),
                key2='loss', ylim2=(0.0, 0.005))

    # WYPRINTOWANIE WAG NEURONÓW
    weights = network.layers[0].get_weights()[0]
    print("\nWAGI:\n")
    print(weights)

    # POBRANIE REZULTATU I OBLICZENIE BŁĘDU
    result = network.predict(dynamic_measurement)
    result = result * 10000

    error_mlp = calculate_mean_squared_err(result, reference, choice="result")
    error_meas = calculate_mean_squared_err(measurement, reference)
    print("Mean square error of the measured values =",
          sum(error_meas) / len(error_meas))
    print("Mean square error of the corrected values =",
          sum(error_mlp) / len(error_mlp))
    print(
        "Arithmetic mean of the input weights [measurement / reference] =", np.mean(weights, axis=1))

    # ZAPISANIE DANYCH DO PLIKU XLSX
    result = pd.DataFrame(result)
    result.to_excel('results.xlsx', engine='xlsxwriter')

    # WYPISANIE WYNIKÓW
    print("\nWYNIKI:\n")
    print(result)
