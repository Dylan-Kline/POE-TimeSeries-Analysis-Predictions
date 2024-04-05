import pandas as pd
import numpy as np

class FeatureEngineer:

    @staticmethod
    def apply_all_features(data: pd.DataFrame):
        '''
            Applies all features in the FeatureEngineer class to the given dataframe
            @ data : pandas dataframe
            return : updated data'''
        data = FeatureEngineer.timeOfLeague_feature(data)
        data = FeatureEngineer.estimated_chaos_features(data)
        data = FeatureEngineer.add_all_lagged_features(data)
        data = FeatureEngineer.rolling_features(data)

        return data
    
    @staticmethod
    def rolling_features(data: pd.DataFrame, local_window_size=5, general_window_size=15, future_context=False):
        '''
          Adds the rolling min, mean, and max for the given window sizes.
          @ data : pandas dataframe to be modified
          @ local_window_size : window size for the local rolling values
          @ general_window_size : window size for the general rolling values
          @ future_context : bool flag that controls whether we include future values 
          in the rolling value calculation or not
          return : modified dataframe 'data' 
        '''
        return data.groupby(by='League').apply(
            FeatureEngineer.apply_rolling,
            local_window_size,
            general_window_size,
            future_context
        ).reset_index(drop=True)
    
    @staticmethod
    def apply_rolling(league : pd.DataFrame, local_window_size = 5, general_window_size = 15, future_context=False):
        '''
            Helper function for @rolling_features method.
            '''
        local_rolling = league['Value'].rolling(local_window_size, min_periods=1, center=future_context)
        general_rolling = league['Value'].rolling(general_window_size, min_periods=1, center=future_context)

        # Local rolling values
        league['rolling_min'] = local_rolling.min()
        league['rolling_mean'] = local_rolling.mean()
        league['rolling_max'] = local_rolling.max()

        # General rolling values
        league['gen_rolling_min'] = general_rolling.min()
        league['gen_rolling_mean'] = general_rolling.mean()
        league['gen_rolling_max'] = general_rolling.max()

        return league
    
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
        
        if 'LeaguePeriod' not in data.columns:
            data = FeatureEngineer.timeOfLeague_feature(data)

        # Add chaos generated per day value
        data['ChaosPerDay'] = data.groupby('League')['DayOfLeague'].apply(FeatureEngineer.calc_chaos_generated_per_day).reset_index(drop=True)

        # Add total chaos at the time of day
        data['TotalChaos'] = data.groupby('League')['ChaosPerDay'].cumsum()
        
        return data

    @staticmethod
    def calc_chaos_generated_per_day(day: int):

        initial_player_count = 99577 # estimated initial player count at league start
        end_player_count = 11445 # estimated end player count at league end
        decay_rate = 0.0265348321 # decay rate of player count over time

        # Player count at the current day and chaos generated for that day per player
        player_count = (initial_player_count - end_player_count) * np.exp(-decay_rate * (day - 1)) + end_player_count
        chaos_gen_rate = FeatureEngineer.calc_chaos_rate(day)

        # Chaos generated for the day
        chaos_for_day = player_count * chaos_gen_rate
        return chaos_for_day

    @staticmethod
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
    def partition_data(league: pd.DataFrame):

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

    @staticmethod
    def add_all_lagged_features(data: pd.DataFrame):
        '''
            Adds three new features to the given dataframe, 'lag_1' - the price one day ago,
            'lag_2' - the price three days ago, 'lag_3' - the price one week ago.
            @ data : pandas dataframe
            return : data with new features.
            '''
        data = data.groupby(by='League').apply(FeatureEngineer.apply_lags).reset_index(drop=True)
        return data

    @staticmethod
    def apply_lags(league_data: pd.DataFrame):
        '''
            Applies the lag features to the given League's data
            '''
        league_data['lag_1'] = FeatureEngineer.add_one_day_lag(league_data['Value']).fillna(0)
        league_data['lag_2'] = FeatureEngineer.add_three_day_lag(league_data['Value']).fillna(0)
        league_data['lag_3'] = FeatureEngineer.add_week_lag(league_data['Value']).fillna(0)
        return league_data

    @staticmethod
    def add_one_day_lag(data: pd.Series):
        '''
            Adds a 'lag_1' feature which is the price one day ago.
            @ data : pandas dataframe
            return : data with new features.
            '''
        return data.shift(1)

    @staticmethod
    def add_three_day_lag(data: pd.Series):
        '''
            Adds a 'lag_2' feature which is the price three day ago.
            @ data : pandas dataframe
            return : data with new features.
            '''
        return data.shift(3)

    @staticmethod
    def add_week_lag(data: pd.Series):
        '''
            Adds a 'lag_3' feature which is the price one week ago.
            @ data : pandas dataframe
            return : data with new features.
            '''
        return data.shift(7)




        
            
            
        