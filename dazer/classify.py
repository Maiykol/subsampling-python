from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from pathlib import Path
import os
import joblib
import settings
from sklearn.inspection import permutation_importance
import glob
import pandas as pd


class Classifier:
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def train_test_random_forest(self, ratio, random_state=101, param_grid={}, n_jobs=-1, model_path=''):
        # random forest
        
        if param_grid == {}:
            param_grid = {
                        'bootstrap': [True],
                        'max_depth': [1, 2, 5, 10, 50, None],
                        'class_weight': ['balanced'],
                        'min_samples_split': [2, 4, 8],
                        'min_samples_leaf': [1, 2, 4, 8],
                        'n_estimators': [10, 100, 250, 500, 750, 1000]
                    }
            
        rf = RandomForestClassifier()
        model = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 1)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        clrep = classification_report(self.y_test, y_pred, target_names=None, output_dict=True)
        
        if model_path:
            Path(model_path).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)

        return model, {'n_samples_train': len(self.X_train), 
                'n_samples_test': len(self.X_test), 
                'accuracy': clrep['accuracy'],
                'f1': clrep['1']['f1-score'],
                'precision': clrep['1']['precision'],
                'recall': clrep['1']['recall'],
                'TNR': clrep['0']['recall'],
                'ratio': ratio,
                'random_state': random_state,
                }
        
        
    def get_feature_weights_random_forests(self, model_paths=[]):
        weights = []
        columns = self.X_train.columns
        for rf_path in  model_paths:
            clf = joblib.load(rf_path)
            data = list(clf.best_estimator_.feature_importances_)
            data.append(rf_path)
            weights.append(data)
            
        columns.append('model_path')
        df = pd.DataFrame(data=weights)

        df['model'] = df['model_path'].map(lambda x: x.split('/')[-1])
        return df
        
        
    def permutation_test_random_forest(self, model_path, ratio, random_state=101):
        # random forest
        model = joblib.load(model_path)
        
        permutation_result = permutation_importance(
            model, self.X_train, self.y_train, n_repeats=10, random_state=random_state, n_jobs=5
        )
        
        return {
            'ratio': ratio,
            'dataset': self.dataset_name,
            'permutation_importances_mean': permutation_result.importances_mean,
            'permutation_importances_std': permutation_result.importances_std
        }