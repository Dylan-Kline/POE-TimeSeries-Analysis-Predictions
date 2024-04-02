import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

class DataVisualization:

    @staticmethod
    def histodist_for_features(data) -> None:
        features = data.select_dtypes(include=['float'])

        for feature in features:
            sns.histplot(data[feature], kde=True, edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.show()
            
    @staticmethod
    def plot_probability_density(data) -> None:
        features = data.select_dtypes(include=['float'])
        
        for feature in features:
            sns.kdeplot(data, bw_adjust=1, label='Probability Density')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title('Probability Density Distribution')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
    @staticmethod
    def boxstrip_plot(data, y_feature, x_feature) -> None:
        '''
            Create a combined box/strip plot for the given y_feature and x_feature.
            @ y_feature : y-axis feature
            @ x_feature : x-axis feature'''
        
        # Create plot figure
        figure = plt.figure(figsize=(15,10))
        
        # Create box plot
        sns.boxplot(x=x_feature, y=y_feature, data=data, palette="pastel", hue=x_feature, legend=False)
        
        # Overlap with stripplot
        sns.stripplot(x=x_feature, y=y_feature, data=data, color='black')
        
        # Adding titles and labels (optional)
        plt.title(f'variable = {str(x_feature)}')
        plt.xlabel(f'{str(x_feature)}')
        plt.ylabel('Exalted Price')

        # Display the plot
        plt.tight_layout()
        plt.show()
        

    @staticmethod
    def visualize_price_all_leagues(data) -> None:
        '''
            Visualizes the price for all leagues in one graph.
            '''
        # Create figure and axis
        figure = plt.figure(figsize=(15, 10))

        # Convert 'Date' column in dataframe to datetime and extract the days
        data['Date'] = pd.to_datetime(data['Date'])

        # Calc the start date of each league
        start_dates = data.groupby('League')['Date'].min().rename('StartDate')

        # Merge the start dates back into the original DataFrame
        data = data.merge(start_dates, on='League')

        data['DayOfLeague'] = (data['Date'] - data['StartDate']).dt.days + 1
        data.to_csv('data/V2.currency.csv', index=False)

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
