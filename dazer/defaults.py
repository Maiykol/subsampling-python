class ParameterGrid:
    random_forest = {
                'bootstrap': [True],
                'max_depth': [1, 2, 5, 10, 50, None],
                'class_weight': ['balanced'],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 4, 8],
                'n_estimators': [10, 100, 250, 500, 750, 1000],
                'random_state': [101]
            }
    
    