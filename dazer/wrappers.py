import numpy as np
import numpy.random as npr
import dazer


def subsample_iterative(df, columns_keep_ratio=[], allowed_deviation=.2, test_size=.2, random_states=[101, 102, 103, 104, 105], attempts=10000, ratios=np.arange(.1, 1.1, .1)):
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
                    df_train_dict[f'{random_state}_{random_state_subsample}_{ratio}'] = subsampler.subsample(
                        subsample_factor=ratio, random_state=random_state_subsample, raise_exception=True)
                    break
                except Exception as e:
                    print(e)
                    i += 1
    return df_test_dict, df_train_dict
