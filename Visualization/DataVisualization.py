import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

plt.style.use('fivethirtyeight')

class DataVisualization:

    @staticmethod
    def histodist_for_features(data: pd.DataFrame) -> None:
        features = data.select_dtypes(include=['float'])

        for feature in features:
            sns.histplot(data[feature], kde=True, edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.show()
            
    @staticmethod
    def plot_probability_density(data: pd.DataFrame) -> None:
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
    def boxstrip_plot(data: pd.DataFrame, y_feature, x_feature) -> None:
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
    def stepped_plot_all(data: pd.DataFrame, x_feature, y_feature, leagues) -> None:
        plot_params = dict(
            color="0.75",
            style=".-",
            markeredgecolor="0.25",
            markerfacecolor="0.25",
            legend=False,
        )

        num_leagues = len(leagues)
        num_columns = 5
        num_rows = math.ceil(num_leagues/num_columns)

        # Create figure and subplots
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(10, num_rows * 2))

        for i, league in enumerate(leagues):
            # Calc row and column index for league plot
            row_idx = i // num_columns
            col_idx = i % num_columns

            # Grab specific league's data
            data_to_plot = data[data['League'] == league]

            # Sort based on date
            data_to_plot.sort_values(by=x_feature)

            # Plot current league's data
            ax = axs[row_idx, col_idx]
            data_to_plot[y_feature].plot(ax=ax, **plot_params)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
            ax.set_title(f'League = {str(league)}', fontsize=6)
            ax.set_ylabel(y_feature, fontsize=6)
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
        
        # If the last row of graph is empty
        if num_leagues % num_columns != 0:
            axs[-1, -1].axis('off')

        # if some subplots are empty
        for ax in axs.flat[num_leagues:]:
            ax.remove()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_price_all_leagues(data: pd.DataFrame) -> None:
        '''
            Visualizes the price for all leagues in one graph.
            '''
        # Create figure and axis
        figure = plt.figure(figsize=(15, 10))

        # Generate a large number of colors
        num_leagues = len(data['League'].unique())
        colors = plt.cm.get_cmap('tab20c_r', num_leagues)
        
        # Grab each leagues data
        leagues = data['League'].unique()

        # Create a line plot for each league 
        for i, league in enumerate(data['League'].unique()):
            
            # filter the data to grab each leagues specific data
            league_data = data[data['League'] == league]

            # Create line plot for league_data
            sns.lineplot(data=league_data, x='DayOfLeague', y='Value', label=league, color=colors(i))
        
        plt.title('Exalted Orb trend per League')
        plt.xlabel('Day of the League')
        plt.ylabel('Value')
        plt.legend(title='League')
        plt.tight_layout()
        plt.show()
