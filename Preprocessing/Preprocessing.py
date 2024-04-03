import pandas as pd
from sklearn.preprocessing import *

class Preprocessing:
    '''
        Collection of preprocessing methods for pandas dataframes.
        '''

    def standardize_data(data: pd.DataFrame) -> pd.DataFrame:
        '''
            Standardizes the features of the given dataframe to be gaussian with a mean of 0.
            @ data : pandas dataframe
            return : standardized dataframe
            '''
        pass

    def z_score_normalization(data: pd.DataFrame) -> pd.DataFrame:
        pass

    def minmax_normalization(data: pd.DataFrame) -> pd.DataFrame:
        pass

