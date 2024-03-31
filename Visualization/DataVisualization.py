import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

class DataVisualization:

    @staticmethod
    def histodist_for_features(data):
        pass

    @staticmethod
    def visualize_price_all_leagues(data):
        '''
            Visualizes the price for all leagues in one graph.
            '''
        # Create figure and axis
        figure = plt.figure(figsize=(15, 10))
        ax = figure.add_subplot(1, 1, 1)

        # Convert 'Date' column in dataframe to datetime and plot using seaborn
        data['Date'] = pd.to_datetime(data['Date'])
        sb.lineplot(data=data, x='Date', y='Value', hue='League', ax=ax)

        plt.tight_layout()
        plt.show()
