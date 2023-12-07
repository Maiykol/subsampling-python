import unittest
import dazer
import pandas as pd
import seaborn as sns
import numpy as np
import tempfile
import shutil
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Testing(unittest.TestCase):


    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        return super().setUp()


    def tearDown(self) -> None:
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        return super().tearDown()


    def test_Classifier_random_forest(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test(
            'rf', param_grid={
                'bootstrap': [True],
                'max_depth': [2],
                'class_weight': ['balanced'],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'n_estimators': [10],
                'random_state': [101]
            }, scoring='f1', cv=3)
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.7879)
        
        
    def test_Classifier_xgboost(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test(
            'xgb', scoring='f1')
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8182)
        
        
    def test_Classifier_mlp(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test(
            'mlp', scoring='f1')
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8485)
        
        
    def test_Classifier_mlp_parameters(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test('mlp', scoring='f1', param_model={
                                                  'solver': 'adam', 'hidden_layer_sizes': (2, 1), 'this_parameter_does_not_exist': None})
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.4545)


    def test_Classifier_gnb(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test('gnb', scoring='f1')
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8788)
    
    
    def test_Classifier_svc(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test('svc', scoring='f1')
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.4545)
        
        
    def test_Classifier_svc_parameters(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test('svc', scoring='f1', param_model={
                                                  'kernel': 'linear', 'C':0.025, 'this_parameter_does_not_exist': None})
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8788)
