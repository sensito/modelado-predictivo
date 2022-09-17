# defina una clase abstracta llamada DatasetSummary con la firma de las funciones en basic.py
from abc import ABC, abstractmethod


class DatasetSummary(ABC):
    
    # defina una función abstracta llamada list_features(self): Conjunto de características cargados para el dataset.
    @abstractmethod
    def list_features(self):
        raise ValueError()
    # defina una función abstracta llamada list_labels(self): Conjunto de etiquetas cargados para el dataset.
    @abstractmethod
    def list_labels(self):
        raise ValueError()
    # defina una función abstracta llamada count_categorical(self): Conteo de características categóricas.
    @abstractmethod
    def count_categorical(self):
        raise ValueError()
    # defina una función abstracta llamada count_numerical(self): Conteo de características numéricas.
    @abstractmethod
    def count_numerical(self):
        raise ValueError()
    # defina una función abstracta llamada statistics(self): Retorna un diccionario con las estadísticas básicas de cada característica.
    @abstractmethod
    def statistics(self):
        raise ValueError()