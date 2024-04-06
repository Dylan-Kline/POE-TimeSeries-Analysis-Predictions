import pandas as pd

class DataProcessing:
    
    @staticmethod
    def train_val_split(df, selected_features, test_league):
        '''
            Splits the data into training and validation sets.
            @ df : pandas dataframe
            @ selected_features : features to be used in training
            @ test_league : League to be used as the testing grounds
            return : training_data, validation_data tuple
            '''
        df_selected = df[selected_features]
        df_train = df_selected[~test_league]
        df_val = df_selected[test_league]
        return df_train, df_val