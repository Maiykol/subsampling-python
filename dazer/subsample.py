import numpy as np
import numpy.random as npr


class Subsampler:
    
    def __init__(self, df_data, column_target, columns_keep_ratio, allowed_deviation):

        df_ratio = df_data[columns_keep_ratio + [column_target]]
        df_ratio = df_ratio.dropna(subset=[column_target])  # remove entries in target column with NaN
        
        self.df_data = df_data  # complete input data
        self.df_ratio = df_ratio  # the dataframe with the columns in which to keep the ratio
        self.column_target = column_target  # the column to predict
        self.allowed_deviation = allowed_deviation  # the allowed deviation from the ratio in the subsampled dataframe in the columns 'columns_keep_ratio' from 0 to 1
        # self.test_size = test_size  
        
        # attributes to be filled by methods
        self._test_df = None  # contains test data, created by 'extract_test' if test_size > 0

        # init steps
        self._preprocess()
        
        
    def _preprocess(self):
        # make all columns numerical
        for col in self.df_ratio:
            if self.df_ratio[col].dtype not in ['float64', 'int64', 'bool']:
                self.df_ratio[col] = self.column_encode_category_as_numerical(self.df_ratio[col])
                
        # normalize columns
        for col in self.df_ratio:
            self.df_ratio[col] = self.column_normalize(self.df_ratio[col])
            
    
    def extract_test(self, test_size=.2, random_state=101):
        """_summary_

        Args:
            test_size (float, optional): percentage of datapoints in test dataset which will be excluded from subsampling. Defaults to .2.
            random_state (int, optional): the random seed. Defaults to 101.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        if type(test_size) not in [int, float] or not 1 >= test_size >= 0:
            raise ValueError('Parameter "test_size" has to be a value in the range from 0 to 1.')
        
        if self.test_size == 0:
            return
        # subsample with equal data distributions
        self._test_df = self.subsample(self.test_size, random_state=random_state)
        # remove test data points from df_data to exclude it in the subsampling process
        self.df_ratio = self.df_ratio[~self.df_ratio.index.isin(self._test_df.index)]
        
        return self._test_df
        
    
    def subsample(self, subsample_factor, random_state=101):
        npr.seed(random_state)
        # subsample each label category inidividually to keep intrinsic ratios
        indices = set()
        for target_category in self.df_ratio[self.column_target].unique():
            df_target_category = self.df_ratio[self.df_ratio[self.column_target]==target_category]

            df_mean_orig = self.df_ratio.mean()
            df_subsampled = df_target_category.sample(frac=subsample_factor, replace=False, random_state=random_state)

            # check for allowed divergence
            for col, col_mean in df_subsampled.drop(self.column_target, axis=1).mean().items():
                if np.abs(df_mean_orig[col] - col_mean) > self.allowed_deviation:
                    raise Exception(f'Could not find subsample with seed {random_state} due to deviation in column {col}.')
                
            indices |= set(df_subsampled.index)

        df_subsampled = self.df_data[self.df_data.index.isin(indices)]

        return df_subsampled
    
    
    @staticmethod
    def column_encode_category_as_numerical(col):
        mapping = {}
        for i, cat in enumerate(col.unique()):
            mapping[cat] = i
        return col.map(mapping)
    
    
    @staticmethod
    def column_normalize(col):
        col_min = col.min()
        col_max = col.max()
        if col_min == col_max == 0:
            # all 0s
            return col
        return (col - col_min) / (col_max - col_min)

