import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    
    # Remove irrelevant features
    df.drop(['Get', 'Pay', 'League', 'StartDate'], axis=1, inplace=True)
    df.info()
    
    ### Normalize and split the data ###
    
    # Grab the numeric and categorical features
    numeric_cols = df.select_dtypes(['float64']).columns
    numeric_cols = numeric_cols.drop('Value')
    categorical_cols = df.select_dtypes(['int64', 'int32', 'datetime64']).columns
    
    # Split the data into training, and testing datasets
    x_examples = df.drop('Value', axis=1)
    y_examples = df['Value']
    
    x_train, x_val, y_train, y_val = train_test_split(x_examples, y_examples, test_size=0.2, random_state=42)
    
    # Normalize training and validation datasets
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train[numeric_cols])
    x_val_scaled = scaler.transform(x_val[numeric_cols])

    # # Convert the scaled arrays back to pandas DataFrame (Retain original index)
    # x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=numeric_cols, index=x_train.index)
    # x_val_scaled_df = pd.DataFrame(x_val_scaled, columns=numeric_cols, index=x_val.index)

    # # Concatenate with categorical columns using the index for correct alignment
    # x_train_full = pd.concat([x_train_scaled_df, x_train[categorical_cols]], axis=1)
    # x_val_full = pd.concat([x_val_scaled_df, x_val[categorical_cols]], axis=1)
    
    # # Change index to date
    # x_train_full.set_index('Date')
    # x_val_full.set_index('Date')

    # # Save training and validation data
    # x_train_full.to_csv('data/train/x_train.csv', index=True)
    # y_train.to_csv('data/train/y_train.csv', index=True)
    # x_val_full.to_csv('data/test/x_val.csv', index=True)
    # y_val.to_csv('data/test/y_val.csv', index=True)
    
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