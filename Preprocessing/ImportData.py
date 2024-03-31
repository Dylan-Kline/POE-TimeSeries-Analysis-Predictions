import numpy as np
import pandas as pd

def data_helper(name, data):
    '''
        '''

def grab_all_leagues_data():
    '''
        Parses through the given datapath for the data in .csv format.
        Returns a dataframe that contains all data from all leagues for a given 
        'GET' item and 'PAY' item. 
        '''
    leagues = ['Betrayal', 'Breach', 'Delirium', 'Delve', 'Heist', 'Legacy',
                'Abyss', 'Bestiary', 'Essence', 'Harbinger', 'Metamorph',
                'Synthesis', 'Incursion', 'Legion', 'Ritual', 'Blight',
                'Harvest', 'Ultimatum', 'Expedition', 'Scourge', 'Archnemesis',
                'Sentinel', 'Kalandra', 'Sanctum', 'Crucible', 'Ancestor']
    
    # Load all data for currency in each league
    df_list = list()
    for league in leagues:
        file_path = "data/" + league + "/" + league + ".currency.csv"
        print(f"Loading {league} currency item data")

        loaded_df = pd.read_csv(file_path)
        df_list.append(loaded_df)

    # Combine seperate league data into one dataframe and save for later use
    unfiltered_df = pd.concat(df_list)
    unfiltered_df.to_csv('data/currency.csv', index=False)

