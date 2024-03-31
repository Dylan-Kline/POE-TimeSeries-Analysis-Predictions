import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualization:

    @staticmethod
    def histodist_for_features(data):
        features = data.columns.values

        for feature in features:
            sns.histplot(data[feature], kde=True, edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.show()

    @staticmethod
    def visualize_price_all_leagues(data):
        '''
            Visualizes the price for all leagues in one graph.
            '''
        # Create figure and axis
        figure = plt.figure(figsize=(10, 10))

        # Convert 'Date' column in dataframe to datetime and extract the days
        data['Date'] = pd.to_datetime(data['Date'])

        # Calc the start date of each league
        start_dates = data.groupby('League')['Date'].min().rename('StartDate')

        # Merge the start dates back into the original DataFrame
        data = data.merge(start_dates, on='League')

        data['DayOfLeague'] = (data['Date'] - data['StartDate']).dt.days + 1
        print(data)

        # Grab each leagues data
        leagues = data['League'].unique()

        # Create a line plot for each league 
        for league in leagues:
            
            # filter the data to grab each leagues specific data
            league_data = data[data['League'] == league]

            # Create line plot for league_data
            sns.lineplot(data=league_data, x='DayOfLeague', y='Value', label=league)
        
        plt.title('Exalted Orb trend per League')
        plt.xlabel('Day of the League')
        plt.ylabel('Value')
        plt.legend(title='League')
        plt.tight_layout()
        plt.show()
