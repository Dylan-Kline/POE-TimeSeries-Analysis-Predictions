import pandas as pd
import numpy as np

class FeatureEngineer:
    
    @staticmethod
    def add_week_of_league(data):
        '''
            Takes a given pandas dataframe and adds a 'weekOfLeague' feature.
            @ data : pandas dataframe with 'DayOfLeague' parameter
            return : data with new feature 'weekOfLeague'
            '''
        data['WeekOfLeague'] = np.ceil(data['DayOfLeague'] / 7).astype(int)
            
            
        