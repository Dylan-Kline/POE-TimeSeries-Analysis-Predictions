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
    def estimated_chaos_features(data: pd.DataFrame) -> pd.DataFrame:
        '''
            Adds an estimated chaos coming into the economy per day
            and estimated amount of chaos in circulation 
            to the given dataframe, provided it includes the 'LeaguePeriod' feature. 
            Otherwise, adds a new feature to indicate the time of league 'start', 'mid', 'end', 
            then uses this to create a chaos creation rate feature.
            @ data : pandas dataframe
            return : chaos engineered dataframe
            '''
        
        if 'LeaguePeriod' not in data.columns():
            data = FeatureEngineer.timeOfLeague_feature(data)

        # Player count is an exponential decay function: count = (init_amount - C) * e ^ (-k * t)) + C
        

    def calc_chaos_rate(day: int):
        
        max_rate = 60 # per day
        growth_rate = 0.17
        base_rate = 4 # per day
        t0 = -np.log((max_rate / base_rate - 1)) / growth_rate
        
        # Calc current chaos rate
        chaos_rate = max_rate / (1 + np.exp(-growth_rate * (day + t0)))
        return chaos_rate
        
        

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




        
            
            
        