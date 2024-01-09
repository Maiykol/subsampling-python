from sklearn.model_selection import GridSearchCV
from sklearn.metrics import max_error, explained_variance_score, r2_score, mean_squared_error
from pathlib import Path
import joblib
import os
import pandas as pd
from dazer import utils


class Regressor:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @staticmethod
    def get_model_from_string(model_string: str):
        if model_string == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor()
        elif model_string == 'xgb':
            from xgboost import XGBRegressor
            return XGBRegressor(objective='reg:squarederror',)
        elif model_string == 'mlp':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor()
        elif model_string == 'svr':
            from sklearn.svm import SVR
            return SVR()

    def train_test(self, model_string: str, param_grid={}, cv=10, n_jobs=-1, model_path='', verbose=1, scoring='max_error'):
        """
        for the scoring options, refer to https://scikit-learn.org/stable/modules/model_evaluation.html
        """
        classifier = self.get_model_from_string(model_string)
        model = self.train(classifier, param_grid=param_grid, cv=cv,
                           n_jobs=n_jobs, verbose=verbose, scoring=scoring)

        y_pred = self.test(model)
        evals = self.eval_pred(y_pred)

        if model_path:
            Path(os.sep.join(model_path.split(os.sep)[
                 :-1])).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)

        return model, evals

    def train(self, model, param_grid={}, cv=10, n_jobs=-1, verbose=1, scoring='f1'):
        model = GridSearchCV(estimator=model, param_grid=param_grid,
                             cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring)
        model.fit(self.X_train, self.y_train)
        return model

    def test(self, model):
        return model.predict(self.X_test)

    def eval_pred(self, y_pred):
       return {'n_samples_train': len(self.X_train),
               'n_samples_test': len(self.X_test),
               'max_error': max_error(self.y_test, y_pred),
               'explained_variance': explained_variance_score(self.y_test, y_pred),
               'r2': r2_score(self.y_test, y_pred),
               'mean_squared_error': mean_squared_error(self.y_test, y_pred, squared=True),
               'root_mean_squared_error': mean_squared_error(self.y_test, y_pred, squared=False)
               }

    def get_feature_weights_random_forests(self, model_paths=[]):
        weights = []
        columns = self.X_train.columns
        for rf_path in model_paths:
            clf = joblib.load(rf_path)
            data = list(clf.best_estimator_.feature_importances_)
            data.append(rf_path)
            weights.append(data)

        columns.append('model_path')
        df = pd.DataFrame(data=weights)

        df['model'] = df['model_path'].map(lambda x: x.split('/')[-1])
        return df

    def classifier_prediction_evaluation(self, models):
        prediction_evaluation_list = []
        for model in models:
            y_pred = model.predict(self.X_test)
            evaluation = utils.evaluate_prediction(self.y_test, y_pred)
            prediction_evaluation_list.append(evaluation)
        return prediction_evaluation_list

    def simulate_random_classifier_prediction_evaluation(self, n_models, random_state=101):

        random_evaluation_list = utils.random_classifier_prediction_evaluation(
            n_models, self.y_test, len(self.X_test), random_state)

        return random_evaluation_list
