import unittest
import dazer
import pandas as pd
import numpy as np
import seaborn as sns
import tempfile
import shutil
from sklearn.datasets import make_classification, make_regression
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
            'xgb', scoring='f1', param_grid={'random_state': [101]})
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8182)
        
        
    def test_Classifier_mlp(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test(
            'mlp', scoring='f1', param_grid={'random_state': [101]})
        print(evaluation['accuracy'])
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8485)
        
        
    def test_Classifier_mlp_parameters(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test('mlp', scoring='f1', param_grid={
                                                  'solver': ['adam'], 'hidden_layer_sizes': (2, 1), 'random_state': [101]})
        print(evaluation['accuracy'])
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.5455)


    def test_Classifier_gnb(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test(
            'gnb', scoring='f1')
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8788)
    
    
    def test_Classifier_svc(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test(
            'svc', scoring='f1', param_grid={'random_state': [101]})
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8788)
        
        
    def test_Classifier_svc_parameters(self):
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=444)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        classifier = dazer.Classifier(pd.DataFrame(
            X_train), y_train, pd.DataFrame(X_test), y_test)
        model, evaluation = classifier.train_test('svc', scoring='f1', param_grid={
                                                  'kernel': ['linear'], 'C': [0.025], 'random_state': [101]})
        self.assertTrue(round(evaluation['accuracy'], 4) == 0.8788)


    def test_Regressor_rf(self):
        X, y = make_regression(n_samples=1000, n_features=10, noise=1, random_state=444)
        target_column = 'y'
        df = pd.DataFrame(X)
        df = df.join(pd.Series(y, name=target_column))

        subsampler = dazer.Subsampler(df, columns_keep_ratio=[target_column], allowed_deviation=.2)

        df_test = subsampler.extract_test(test_size=.2, random_state=101)
        df_train = subsampler.subsample(subsample_factor=.1, random_state=101)

        y_test = df_test[target_column]
        X_test = df_test.drop([target_column], axis=1)

        y_train = df_train[target_column]
        X_train = df_train.drop([target_column], axis=1)

        regressor = dazer.Regressor(X_train, y_train, X_test, y_test)
        model, evaluation = regressor.train_test('rf', scoring='r2')
        
        self.assertTrue(round(evaluation['r2'], 4) == 0.7071)
        
        
    def test_Regressor_svr(self):
        X, y = make_regression(
            n_samples=1000, n_features=10, noise=1, random_state=444)
        target_column = 'y'
        df = pd.DataFrame(X)
        df = df.join(pd.Series(y, name=target_column))

        subsampler = dazer.Subsampler(df, columns_keep_ratio=[
                                      target_column], allowed_deviation=.2)

        df_test = subsampler.extract_test(test_size=.2, random_state=101)
        df_train = subsampler.subsample(subsample_factor=.1, random_state=101)

        y_test = df_test[target_column]
        X_test = df_test.drop([target_column], axis=1)

        y_train = df_train[target_column]
        X_train = df_train.drop([target_column], axis=1)

        regressor = dazer.Regressor(X_train, y_train, X_test, y_test)
        model, evaluation = regressor.train_test('svr', scoring='r2')

        self.assertTrue(round(evaluation['explained_variance'], 4) == 0.0239)


    def test_one_class(self):
        target_column = 'y'

        df = sns.load_dataset('penguins', data_home=self.test_dir)
        df = df.dropna()
        df[target_column] = df['species'] == 'Adelie'
        
        subsampler = dazer.Subsampler(df, ['body_mass_g', 'y'], .07, True)

        df_test = subsampler.extract_test(.2, random_state=2)
        df_train = subsampler.subsample(.4, random_state=3)
        
        y_test = df_test[target_column]
        X_test = df_test.drop([target_column], axis=1)

        y_train = df_train[target_column]
        X_train = df_train.drop([target_column], axis=1)

        classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
        
        model, evaluation = classifier.train_test('rf', scoring='f1')
        
        self.assertTrue(evaluation['recall'] == 1.0)
