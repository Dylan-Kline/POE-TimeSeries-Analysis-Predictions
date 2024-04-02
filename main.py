import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *
from Visualization.DataVisualization import *
from Preprocessing.FeatureEngineer import *

def main():
    leagues = ['Affliction', 'Betrayal', 'Breach', 
                                                  'Delirium', 'Delve', 'Heist', 'Legacy',
                    'Abyss', 'Bestiary', 'Essence', 'Harbinger', 'Metamorph',
                    'Synthesis', 'Incursion', 'Legion', 'Ritual', 'Blight',
                    'Harvest', 'Ultimatum', 'Expedition', 'Scourge', 'Archnemesis',
                    'Sentinel', 'Kalandra', 'Sanctum', 'Crucible', 'Ancestor']
    data_path = "data/currency.csv"
    df = pd.read_csv("data/currency.csv")

    # Convert date feature to standard datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df.info()

    # Remove irrelevant features
    df.drop(['Get', 'Pay'], axis=1, inplace=True)
    df.info()

    # Data Visualization of data

    # For each league display the stepped plot
    # for league in leagues:
    league = 'Affliction'
    DataVisualization.stepped_plot(df, 'Date', 'Value', league)
    #DataVisualization.visualize_price_all_leagues(df)
    #DataVisualization.histodist_for_features(df)
    #DataVisualization.plot_probability_density(df)
    #DataVisualization.boxstrip_plot(df, 'Value', 'DayOfLeague')
    #DataVisualization.boxstrip_plot(df, 'Value', 'WeekOfLeague')
    #DataVisualization.boxstrip_plot(df, 'Value', 'Confidence')

main()