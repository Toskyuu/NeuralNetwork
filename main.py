import pandas as pd
import tensorflow as tf
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from glob import glob




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


def calculate_mean_squared_err(measurement: pd.DataFrame, reference: pd.DataFrame, choice=None):
    reference = reference.values.tolist()
    mse = []
    measurement = measurement.tolist() if choice == "result" else measurement.values.tolist()
    for i in range(len(measurement)):
        value = mean_squared_error([measurement[i][0], measurement[i][1]], [
            reference[i][0], reference[i][1]])
        mse.append(math.sqrt(value))
    return np.sort(mse) * 10000


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
    network.add(tf.keras.layers.Dense(10, activation='relu'))
    network.add(tf.keras.layers.Dense(5, activation='elu'))
    network.add(tf.keras.layers.Dense(2, activation='selu'))


    # KOMPILACJA SIECI PRZY POMOCY OPTYMALIZATORA "ADAM"
    network.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy'])

    # STWORZENIE EARLYSTOPPING, KTÓRY ZATRZYMA PROGRAM, JEŚLI TEN PRZESTANIE SIĘ UCZYĆ
    early_stop = EarlyStopping(monitor='loss',
                              patience=3,
                              min_delta=0.0000001,
                              verbose=1)
    # NAUCZANIE SIECI PRZY POMOCY DANYCH STATYCZNYCH
    network.fit(measurement, reference,
                          epochs=100,
                          verbose=1,
                          callbacks=[early_stop])

    # WYPRINTOWANIE WAG NEURONÓW
    print("\nWAGI:\n")
    for x in range(0, 3):
       print("\nWarstwa nr ", x+1, "\n")
       print(network.layers[x].get_weights()[0])


    # WYKONANIE PREDYKCJI I OBLICZENIE BŁĘDU
    result = network.predict(dynamic_measurement)
    result = result
    error_mlp = calculate_mean_squared_err(result, dynamic_reference,
                                           choice="result")
    error_meas = calculate_mean_squared_err(dynamic_measurement, dynamic_reference)

    # ZAPISANIE DANYCH DO PLIKU XLSX
    error_mlp = pd.DataFrame(error_mlp)
    error_mlp.to_excel("results/dis_error.xlsx", engine='xlsxwriter')

    #NARYSOWANIE WYKRESU DYSTRYBUNATY BŁĘDU DANYCH
    for errors, label in zip([error_mlp, error_meas], ["Dane filtrowane", "Dane niefiltrowane"]):
        x = 1. * np.arange(len(errors)) / (len(errors) - 1)
        plt.plot(errors, x, label=label)
    plt.legend()
    plt.xlim(0, 2000)
    plt.ylim(0, 1)
    plt.savefig("results/dis_error.jpg")
    plt.show()

