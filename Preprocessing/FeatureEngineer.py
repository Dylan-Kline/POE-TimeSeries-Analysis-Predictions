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

    @staticmethod
    def estimated_chaos_rate(data: pd.DataFrame) -> pd.DataFrame:
        '''
            Adds a chaos creation rate feature to the given dataframe, provided it includes the time of league
            feature. Otherwise, adds a new feature to indicate the time of league 'start', 'mid', 'end', then uses
            this to create a chaos creation rate feature.
            @ data : pandas dataframe
            return : chaos engineered dataframe
            '''
        

    @staticmethod
    def timeOfLeague_feature(data: pd.DataFrame) -> pd.DataFrame:
        '''
            Adds a time of league feature to the given dataframe, which has values 'start', 'mid', 'end'.
            '''
        return data.groupby('League').apply(FeatureEngineer.partition_data).reset_index(drop=True)


    @staticmethod
    def partition_data(league):

        # make sure league data is sorted by DayOfLeague, from start to end of league
        league = league.sort_values(by='DayOfLeague')

        # Grab ending day of league, since it is the same as total length of league
        end = league['DayOfLeague'].iloc[-1]

        # Parition the league timespan into 3
        start_index = end // 3
        middle_index = start_index * 2

        league['LeaguePeriod'] = league['DayOfLeague'].map(lambda x: 0 if x < start_index
                                                  else 1 if x < middle_index
                                                  else 2)
        
        return league




        
            
            
        