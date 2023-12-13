import pickle
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
    """
    Verilen DataFrame üzerinden çizgi grafiği çizer ve kaydeder.
    
    :param df: Çizim için kullanılacak pandas DataFrame.
    :param chart_name: Grafiğin kaydedileceği dosya adı.
    :param title: Grafiğin başlığı.
    :param x_axis_label: X ekseni için etiket.
    :param y_axis_label: Y ekseni için etiket.
    """
    plt.figure(figsize=(15, 8))
    plt.plot(df)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.savefig(chart_name)
    plt.close()

def save_model(model, filename):
    """
    Verilen modeli belirtilen dosya adıyla kaydeder.

    :param model: Kaydedilecek model.
    :param filename: Modelin kaydedileceği dosya adı.
    """
    model.save(filename)

def load_trained_model(filename):
    """
    Belirtilen dosya adından eğitilmiş bir model yükler.

    :param filename: Yüklenecek modelin dosya adı.
    :return: Yüklenen model.
    """
    return load_model(filename)

def save_to_pickle(data, filename):
    """
    Verilen veriyi pickle formatında kaydeder.

    :param data: Kaydedilecek veri.
    :param filename: Dosya adı.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    """
    Pickle formatında kaydedilmiş veriyi yükler.

    :param filename: Yüklenmek istenen dosyanın adı.
    :return: Yüklenen veri.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)
