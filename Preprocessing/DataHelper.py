import numpy as np
import pandas as pd

class DataHelper:

    @staticmethod
    def grab_all_leagues_currency_data(leagues = ['Affliction', 'Betrayal', 'Breach', 
                                                  'Delirium', 'Delve', 'Heist', 'Legacy',
                    'Abyss', 'Bestiary', 'Essence', 'Harbinger', 'Metamorph',
                    'Synthesis', 'Incursion', 'Legion', 'Ritual', 'Blight',
                    'Harvest', 'Ultimatum', 'Expedition', 'Scourge', 'Archnemesis',
                    'Sentinel', 'Kalandra', 'Sanctum', 'Crucible', 'Ancestor']):
        '''
            Parses through the given datapath for the data in .csv format.
            Returns a dataframe that contains all data from all leagues for a given 
            'GET' item and 'PAY' item. 
            '''
        
        # Load all data for currency in each league
        df_list = list()
        for league in leagues:
            file_path = "data/" + league + "/" + league + ".currency.csv"
            print(f"Loading {league} currency item data")

            loaded_df = pd.read_csv(file_path, delimiter=";")
            df_list.append(loaded_df)

        # Combine seperate league data into one dataframe and save for later use
        unfiltered_df = pd.concat(df_list)
        unfiltered_df.to_csv('data/currency.csv', index=True)
        return unfiltered_df

    @staticmethod
    def update_data(csv_path, new_data_path):
        '''
            Updates a given csv file with the new data received.
            @ csv_path : path to csv file to update
            @ new_data : path to csv file to grab new data from
            '''
        if new_data_path is None or csv_path is None:
            return None
        
        df_to_update = pd.read_csv(csv_path, delimiter=";")
        df_new = pd.read_csv(new_data_path, delimiter=";")
        df = pd.concat([df_to_update, df_new])
        df.to_csv('data/currency.csv', index=False)
        return df

    @staticmethod
    def extract_exalt_and_divine(dataframe):
        cond1 = (dataframe['Get'] == 'Exalted Orb') & (dataframe['Date'] < '2022-08-19')
        cond2 = (dataframe['Get'] == 'Divine Orb') & (dataframe['Date'] > '2022-08-19')
        dataframe = dataframe[cond1 | cond2]
        return dataframe
