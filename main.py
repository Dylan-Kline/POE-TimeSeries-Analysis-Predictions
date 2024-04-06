import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing.DataHelper import *
from Visualization.DataVisualization import *
from Preprocessing.FeatureEngineer import *
from Preprocessing.Preprocessing import *
from sklearn.preprocessing import *
from DataProcessing import *

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
    df = FeatureEngineer.make_multi_target(df)
    print(df)
    
    # Remove irrelevant features
    df.drop(['Get', 'Pay', 'StartDate'], axis=1, inplace=True)
    df.info()
    
    ### Normalize and split the data ###
    
    # Grab the numeric and categorical features
    y_features = ['y_step_1', 'y_step_2', 'y_step_3', 'y_step_4', 'y_step_5']
    numeric_cols = df.select_dtypes(['float64']).columns
    numeric_cols = numeric_cols.drop(y_features)
    numeric_cols = numeric_cols.drop('Value')
    categorical_cols = df.select_dtypes(['int64', 'int32', 'datetime64']).columns.to_list()
    if 'Value' in df.columns:
        categorical_cols.append('Value')
    
    # Split the data into training, and testing dataset
    input_features = df.columns.drop(y_features)
    output_features = y_features.copy()
    output_features.append('Date')
    
    x_train, x_val, y_train, y_val = DataProcessing.split(df, input_features, output_features, 'Ultimatum')
    
    # Scale the training and validation datasets (only numeric columns)
    scaler = MinMaxScaler()
    x_train_scaled_numeric = scaler.fit_transform(x_train[numeric_cols])
    x_val_scaled_numeric = scaler.transform(x_val[numeric_cols])
    
    # Convert the scaled arrays back to pandas DataFrames
    x_train_scaled_numeric_df = pd.DataFrame(x_train_scaled_numeric, columns=numeric_cols, index=x_train.index)
    x_val_scaled_numeric_df = pd.DataFrame(x_val_scaled_numeric, columns=numeric_cols, index=x_val.index)

    # Concatenate with the categorical columns and datetime columns
    x_train_full = pd.concat([x_train_scaled_numeric_df, x_train[categorical_cols]], axis=1)
    x_val_full = pd.concat([x_val_scaled_numeric_df, x_val[categorical_cols]], axis=1)
    
    # Sorting by 'Date' column
    x_train_full = x_train_full.sort_values(by='Date')
    y_train = y_train.sort_values(by='Date')
    x_val_full = x_val_full.sort_values(by='Date')
    y_val = y_val.sort_values(by='Date')

    # Save to CSV
    x_train_full.to_csv('data/train/x_train_scaled.csv', index=False)
    y_train.to_csv('data/train/y_train.csv', index=False)
    x_val_full.to_csv('data/test/x_val_scaled.csv', index=False)
    y_val.to_csv('data/test/y_val.csv', index=False)
    
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