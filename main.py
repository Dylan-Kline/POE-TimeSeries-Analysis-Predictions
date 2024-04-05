import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *
from Visualization.DataVisualization import *
from Preprocessing.FeatureEngineer import *
from Preprocessing.Preprocessing import *

def main():
    leagues = ['Affliction', 'Betrayal', 'Breach', 
                                                  'Delirium', 'Delve', 'Heist', 'Legacy',
                    'Abyss', 'Bestiary', 'Essence', 'Harbinger', 'Metamorph',
                    'Synthesis', 'Incursion', 'Legion', 'Ritual', 'Blight',
                    'Harvest', 'Ultimatum', 'Expedition', 'Scourge', 'Archnemesis',
                    'Sentinel', 'Kalandra', 'Sanctum', 'Crucible', 'Ancestor']
    
    data_path = "data/currency.csv"
    df = pd.read_csv(data_path)

    # Convert date feature to standard datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df.info()

    # Remove irrelevant features
    df.drop(['Get', 'Pay'], axis=1, inplace=True)
    df.info()
    
    # quick check for null and nan values
    print(df.isna())
    print(df.isnull())

    # Remove outliers from data to see if they affect model accuracy negatively

    ### Feature Engineering ###
    df = FeatureEngineer.apply_all_features(df)
    pd.set_option('display.max_rows', None)
    print(df[df['League'] == 'Ultimatum'])
    
    # Data Visualization of data #

    #DataVisualization.boxstrip_plot(df, 'Value', 'League')
    #DataVisualization.stepped_plot_all(df, 'Date', 'Value', leagues)
    #DataVisualization.visualize_price_all_leagues(df)
    #DataVisualization.histodist_for_features(df)
    #DataVisualization.plot_probability_density(df)
    #DataVisualization.boxstrip_plot(df, 'Value', 'DayOfLeague')
    #DataVisualization.boxstrip_plot(df, 'Value', 'WeekOfLeague')
    #DataVisualization.boxstrip_plot(df, 'Value', 'Confidence')
    
    # Test functions
    #FeatureEngineer.calc_chaos_rate(100)

main()