import numpy as np
import pandas as pd

class ImportData:

    @staticmethod
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
        
        print(len(leagues))

ImportData.grab_all_leagues_data()