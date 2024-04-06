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
    
    @staticmethod
    def label_encoder(data: pd.DataFrame, labels: list, column_name: str) -> pd.DataFrame:
        '''
            Encodes the values in column 'column_name' in the dataframe 'data'.
            @ data : pandas dataframe
            @ labels : list of class labels
            @ column_name : feature column to encode
            return : updated dataframe 'data'
            '''
        label_encoder = LabelEncoder()
        
        # ensure list contains values
        if len(labels) > 0:
            label_encoder.fit(labels)
        else:
            print('list cannot be empty for encoding.')
            return
        
        # transform values in dataframe to their corresponding class labels 
        data[column_name] = label_encoder.transform(data[column_name])
        
        return data

