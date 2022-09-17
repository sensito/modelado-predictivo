#__init__(self, filepath, features=None, labels=None): Constructor que recibirá tres parámetros:
# filepath: Ruta del fichero CSV que contiene los datos.
# features: Lista con los nombres de las columnas que contienen las características.
# labels: Lista con los nombres de las columnas que contienen las etiquetas.
# Para todos los casos que deberían enviar un error devuelva un ValueError 
import pandas as pd
import numpy as np
#heredar de la clase abstracta DatasetSummary en base.py
from .base import DatasetSummary
class TabularDatasetSummary(DatasetSummary): 
    def __init__(self, filepath, features=None, labels=None):
        try:
            if features is not None and labels is not None:
                # features: Lista que contendrá el nombre de las características que se deseen cargar. En caso de que sea None, cargar todas las características.
                # labels: Lista que contendrá el nombre de las características que serán usadas como etiquetas en el dataset. En caso de que sea None, se asumirá que no existe una etiqueta. Todas las columnas especificadas aquí deberás ser excluidas de lalista de features.
   
                self.features  = features
                self.labels = labels
                index = labels - features
                self.data = pd.read_csv(filepath, index_col=index, names=features)
            elif features is not None:
                self.features = features
                self.data = pd.read_csv(filepath, names = features)
            elif labels is not None:
                self.labels = labels
                self.data = pd.read_csv(filepath, index_col=labels)
            else:
                self.data = pd.read_csv(filepath)
        except:
            raise ValueError("Error")   
             
    #Set[string] list_features(self): Conjunto de características cargados para el dataset.
    def list_features(self):
        try:
            return set(self.data.columns)
        except:
            raise ValueError("No hay features")
    #Set[string] list_labels(self): Conjunto de etiquetas cargados para el dataset.
    def list_labels(self):
        try:
            return set(self.data.index)
        except:
            raise ValueError("No hay labels")
    #Integer count_categorical(self): Conteo de características categóricas.
    def count_categorical(self):
        try:
            return self.data[self.data.columns].select_dtypes(include=['object']).shape[1]
        except:
            raise ValueError("No hay features")
    #Integer count_numerical(self): Conteo de características numéricas.
    def count_numerical(self):
        try:
            return self.data[self.data.columns].select_dtypes(include=['float64', 'int64']).shape[1]
        except:
            raise ValueError("No hay features")
        
    def statistics(self):
        try: 
            stats = {}
            for feature in self.data.columns:
                stats[feature] = {}
                if self.data[feature].dtype == 'object':
                    stats[feature]['type'] = 'categorical'
                    stats[feature]['mode'] = self.data[feature].mode()[0]
                    stats[feature]['median'] = None
                    stats[feature]['std'] = None
                else:
                    stats[feature]['type'] = 'numerical'
                    stats[feature]['mean'] = self.data[feature].mean()
                    stats[feature]['mode'] = self.data[feature].mode()[0]
                    stats[feature]['median'] = self.data[feature].median()
                    stats[feature]['std'] = self.data[feature].std()
                stats[feature]['n_null'] = self.data[feature].isnull().sum()
                stats[feature]['n_total'] = self.data[feature].shape[0]
            return stats
        except:
            raise ValueError("Error")
    
    def histogram(self, feature, bins=10):
        try:
            if self.data[feature].dtype == 'object':
                return self.data[feature].value_counts().index, self.data[feature].value_counts().values
            else:
                return np.histogram(self.data[feature], bins=bins)
        except:
            raise ValueError("Error") 