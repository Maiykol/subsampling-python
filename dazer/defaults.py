class ParameterGridClassifier:
    rf = {
                'bootstrap': [True],
                'max_depth': [1, 2, 5, 10, 50, None],
                'class_weight': ['balanced'],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 4, 8],
                'n_estimators': [10, 100, 250, 500, 750, 1000],
                'random_state': [101]
            }
    svc = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    mlp = {
        'hidden_layer_sizes': [(10, 5), (100,), (100, 50,)],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'adam'],
        'random_state': [101],
        'alpha': [1e-5],
        'learning_rate_init': [0.001, 0.01, 0.0001]
    }


class ParameterGridRegressor:
    rf = {
        'bootstrap': [True],
        'max_depth': [1, 2, 5, 10, 50, None],
        'criterion': ["squared_error", "absolute_error"],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4, 8],
        'n_estimators': [10, 100, 250, 500, 750, 1000],
        'random_state': [101]
    }
    svr = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    mlp = {
        'hidden_layer_sizes': [(10, 5), (100,), (100, 50,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['lbfgs', 'adam'],
        'random_state': [101],
        'alpha': [1e-5],
        'learning_rate_init': [0.001, 0.01, 0.0001]
    }
