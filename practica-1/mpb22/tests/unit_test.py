# -*- coding: utf-8 -*-
# Path: modelado-predictivo/practica-1/tests/unittest.py

#Se puede ejecutar con el comando python -m unittest unit_test.py -v para ver los detalles de los test
import sys
sys.path.append("..") 

import unittest
from src.mpb22.eda.tabular import TabularDatasetSummary

class TestTabularDatasetSummary(unittest.TestCase):
    def setUp(self):
        self.dataset = TabularDatasetSummary('./data/heart.csv')
#test para la funcion list_features
    def test_list_features(self):
        self.assertEqual(self.dataset.list_features(), {'exang', 'thal', 'oldpeak', 'cp', 'trestbps', 'slope', 'age', 'fbs', 'thalach', 'target', 'restecg', 'chol', 'sex', 'ca'})
    #test funcion list_labels
    def test_list_labels(self):
        self.assertEqual(self.dataset.list_labels(), 0)
    #test funcion count_categorical = 0
    def test_count_categorical(self):
        self.assertEqual(self.dataset.count_categorical(), 0)
    #test funcion count_numerical = 14
    def test_count_numerical(self):
        self.assertEqual(self.dataset.count_numerical(), 14)
    #test funcion statistics
    def test_statistics(self):
        self.assertEqual(self.dataset.statistics(), {'age': {'type': 'numerical', 'mean': 54.43414634146342, 'mode': 58, 'median': 56.0, 'std': 9.072290233244281, 'n_null': 0, 'n_total': 1025}, 'sex': {'type': 'numerical', 'mean': 0.6956097560975609, 'mode': 1, 'median': 1.0, 'std': 0.4603733241196503, 'n_null': 0, 'n_total': 1025}, 'cp': {'type': 'numerical', 'mean': 0.9424390243902439, 'mode': 0, 'median': 1.0, 'std': 1.0296407436458572, 'n_null': 0, 'n_total': 1025}, 'trestbps': {'type': 'numerical', 'mean': 131.61170731707318, 'mode': 120, 'median': 130.0, 'std': 17.516718005376408, 'n_null': 0, 'n_total': 1025}, 'chol': {'type': 'numerical', 'mean': 246.0, 'mode': 204, 'median': 240.0, 'std': 51.59251020618206, 'n_null': 0, 'n_total': 1025}, 'fbs': {'type': 'numerical', 'mean': 0.14926829268292682, 'mode': 0, 'median': 0.0, 'std': 0.3565266897271594, 'n_null': 0, 'n_total': 1025}, 'restecg': {'type': 'numerical', 'mean': 0.5297560975609756, 'mode': 1, 'median': 1.0, 'std': 0.5278775668748926, 'n_null': 0, 'n_total': 1025}, 'thalach': {'type': 'numerical', 'mean': 149.11414634146342, 'mode': 162, 'median': 152.0, 'std': 23.005723745977196, 'n_null': 0, 'n_total': 1025}, 'exang': {'type': 'numerical', 'mean': 0.33658536585365856, 'mode': 0, 'median': 0.0, 'std': 0.4727723760037095, 'n_null': 0, 'n_total': 1025}, 'oldpeak': {'type': 'numerical', 'mean': 1.0715121951219524, 'mode': 0.0, 'median': 0.8, 'std': 1.175053255150173, 'n_null': 0, 'n_total': 1025}, 'slope': {'type': 'numerical', 'mean': 1.3853658536585365, 'mode': 1, 'median': 1.0, 'std': 0.6177552671745906, 'n_null': 0, 'n_total': 1025}, 'ca': {'type': 'numerical', 'mean': 0.7541463414634146, 'mode': 0, 'median': 0.0, 'std': 1.0307976650242847, 'n_null': 0, 'n_total': 1025}, 'thal': {'type': 'numerical', 'mean': 2.32390243902439, 'mode': 2, 'median': 2.0, 'std': 0.620660238051028, 'n_null': 0, 'n_total': 1025}, 'target': {'type': 'numerical', 'mean': 0.5131707317073171, 'mode': 1, 'median': 1.0, 'std': 0.5000704980788051, 'n_null': 0, 'n_total': 1025}})
    #test funcion histogram
    def test_histogram(self):
        self.assertEqual(str(self.dataset.histogram('age')), '(array([29. , 33.8, 38.6, 43.4, 48.2, 53. , 57.8, 62.6, 67.4, 72.2]), array([  4,  39, 109, 125, 120, 205, 219, 149,  46,   9]))')