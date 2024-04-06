import pandas as pd

class DataProcessing:
    
    @staticmethod
    def split(df, input_features, output_features, test_league):
        '''
            return : x_train, x_val, y_train, y_val
            '''
        x_train, x_val = DataProcessing.train_val_split(df, input_features, test_league)
        y_train, y_val = DataProcessing.train_val_split(df, output_features, test_league)
        
        DataProcessing.print_train_val_stats(x_train, x_val)
        DataProcessing.print_train_val_stats(y_train, y_val)
        
        return x_train, x_val, y_train, y_val
    
    @staticmethod
    def train_val_split(df, selected_features, test_league):
        '''
            Splits the data into training and validation sets.
            @ df : pandas dataframe
            @ selected_features : features to be used in training
            @ test_league : League to be used as the testing grounds
            return : training_data, validation_data tuple
            '''
            
        # Ensure the 'League' column is one of the selected features
        if 'League' not in selected_features:
            selected_features.append('League')
        
        df_selected = df[selected_features]
        df_train = df_selected[df_selected['League'] != test_league].drop('League', axis=1)
        df_val = df_selected[df_selected['League'] == test_league].drop('League', axis=1)
        
        return df_train, df_val
    
    @staticmethod
    def print_train_val_stats(train,val):
        print('shapes: ', train.shape, val.shape)
        print(train.head())
        print(val.head())
        train.info()
        val.info()