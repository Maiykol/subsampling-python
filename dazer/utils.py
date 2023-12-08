import joblib
import numpy.random as npr
import numpy as np
import dazer


def subsample_iterative(df, columns_keep_ratio=[], allowed_deviation=.2, test_size=.2, random_states=[101, 102, 103, 104, 105], attempts=10000, ratios=np.arange(0, 1, .1)):
    """
     Iteratively subsamples a data frame to test data. The subsample is done by randomly choosing a subset of columns from the data frame and then re - sampling the data with a variety of random numbers.
     
     Args:
     	 df: Data frame to be subsampled.
     	 columns_keep_ratio: List of column names to keep ratio is ignored.
     	 allowed_deviation: Allowed deviation is used to determine the probability of a column being considered to be significant.
     	 test_size: Number of samples to be tested.
     	 random_states: List of random numbers to use for the subsampling.
     	 attempts: Number of attempts to subsample before giving up.
     	 ratios: Numpy array of ratios to use for each subsample.
     
     Returns: 
     	 Two dicts with DataFrame values, the first one containing test data and the second one training data. The keys containg information about ratios and random seeds. If there is an error in the data a ValueError is raised
    """
    df_test_dict = {}
    df_train_dict = {}
    # Generate a random state for each of the random states.
    for random_state in random_states:
        subsampler = dazer.Subsampler(
            df, columns_keep_ratio=columns_keep_ratio, allowed_deviation=allowed_deviation)
        npr.seed(random_state)
        i = 0
        # This function will attempt to extract test data from the dataset.
        while i < attempts:
            random_state_subsample = npr.randint(1, 999999999)
            try:
                df_test = subsampler.extract_test(
                    test_size=test_size, random_state=random_state_subsample)
                # Check if df_test is None.
                if df_test is None:
                    raise Exception()
                df_test_dict[f'{random_state}_{random_state_subsample}'] = df_test
                break
            except Exception as e:
                print(e)
                i += 1
        
        # Check if df_test is None.
        if df_test is None:
            raise Exception()
                    
        # This function generates a number of random samples for each ratio.
        for ratio in ratios:
            # If ratio 0 continue to do so.
            if ratio == 0:
                continue
            ratio = round(ratio, 4)
            
            npr.seed(random_state)
            i = 0
            # This function will generate a 'attempts' number of times a random number of attempts.
            while i < attempts:
                random_state_subsample = npr.randint(1, 999999999)
                try:
                    df_train_dict[f'{random_state}_{random_state_subsample}_{ratio}'] = subsampler.subsample(subsample_factor=ratio, random_state=random_state_subsample, raise_exception=True)
                    break
                except Exception as e:
                    print(e)
                    i += 1
    return df_test_dict, df_train_dict


def load_models(paths):
    """
     Load models from a list of paths. This is a convenience function for use with joblib. load ()
     
     Args:
     	 paths: A list of paths to load models from.
     
     Returns: 
     	 A list of : class : list objects corresponding to the loaded models.
    """
    models = []
    # Load jobs from paths.
    for path in paths:
        models.append(joblib.load(path))
    return models


def evaluate_prediction(y, y_pred):
    """
     Evaluate the prediction. This is a helper function for predict_proba. It takes the y values and the predicted values and returns a dictionary that maps the values to their confidence scores.
     
     Args:
     	 y: A list of values to be used for evaluation.
     	 y_pred: A list of predicted values. Each value should be 1 or 0.
     
     Returns: 
     	 A dictionary with the following keys : tp fp fn n_tp n_fp n_fn tnr fn
    """
    n_tp = 0
    n_tn = 0
    n_fp = 0
    n_fn = 0
    for iy, iy_pred in zip(y, y_pred):
        if iy == iy_pred:
            if iy == 1:
                n_tp += 1
            else:
                n_tn += 1
        else:
            if iy == 1:
                n_fn += 1
            else:
                n_fp += 1
    
    tpr = n_tp/(n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    fpr = n_fp/(n_fp + n_tn) if (n_fp + n_tn) > 0 else 0
    tnr = n_tn/(n_tn + n_fp) if (n_tn + n_fp) > 0 else 0
    fnr = n_fn/(n_fn + n_tn) if (n_fn + n_tn) > 0 else 0
    return {'tp': tpr, 'fp': fpr, 'tn': tnr, 'fn': fnr, 'n_tp': n_tp, 'n_tn': n_tn, 'n_fn': n_fn, 'n_fp': n_fp}
        
        
def random_classifier_prediction_evaluation(n_models, y_test, n_X, random_state=101):
    """
     Generate a list of predictions for each model. This is useful for testing the performance of different classifiers
     
     Args:
     	 n_models: Number of models to generate
     	 y_test: Test labels for each model
     	 n_X: Number of data points to use in evaluation
     	 random_state: Random state to use for npr. seed
     
     Returns: 
     	 List of evaluations for each model ( prediction_evaluation_list ). Each evaluation is a list
    """
    npr.seed(random_state)
    prediction_evaluation_list = []
    # This function evaluates the prediction of the model.
    for _ in range(n_models):
        y_random = [npr.random() > 0.5 for _ in range(n_X)]
        evaluation = evaluate_prediction(y_test, y_random)
        prediction_evaluation_list.append(evaluation)
    return prediction_evaluation_list
