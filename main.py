import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *
from Visualization.DataVisualization import *
from Preprocessing.FeatureEngineer import *
from Preprocessing.Preprocessing import *
from sklearn.preprocessing import *

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
    df.drop(['Get', 'Pay', 'Date', 'StartDate'], axis=1, inplace=True)
    df.info()
    
    # quick check for null and nan values
    if 'True' in df.isna():
        print('na values in df')
    
    if 'True' in df.isnull():
        print('null values in df')

    # Remove outliers from data to see if they affect model accuracy negatively
    
    ### Preprocessing ###
    
    # encode the 'Confidence' column
    column_name = 'Confidence'
    class_labels = df[column_name].unique()
    df = Preprocessing.label_encoder(df, class_labels, column_name=column_name)
    
    ### Feature Engineering ###
    
    df = FeatureEngineer.apply_all_features(df)
    print(df)
    
    ### Normalize the numerical data features for later modeling ###
    
    # Grab the numeric and categorical features
    numeric_cols = df.select_dtypes(['float64']).columns
    categorical_cols = df.select_dtypes(['int64']).columns
    
    # normalize numerical data only
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(df_normalized)
    
    
    # Data Visualization of data #

    #DataVisualization.plot_autocorr_all_leagues(df, 1000, column_name='Value')
    #DataVisualization.boxstrip_plot(df, 'Value', 'League')
    #DataVisualization.stepped_plot_all(df, 'Date', 'Value', leagues)
    #DataVisualization.visualize_price_all_leagues(df)
    #DataVisualization.histodist_for_features(df)
    #DataVisualization.plot_probability_density(df)
    #DataVisualization.boxstrip_plot(df, 'Value', 'DayOfLeague')
    #DataVisualization.boxstrip_plot(df, 'Value', 'WeekOfLeague')
    #DataVisualization.boxstrip_plot(df, 'Value', 'Confidence')
    
main()