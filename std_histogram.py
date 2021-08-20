# Author: Benoit DELORME
# Mail: delormebenoit211@gmail.com
# Creation date: 26/06/2021
# Main objective: provide an IT version of the histogram, as described
# by Dr. Ishikawa in its book 'Guide for Quality Control, 1968'


# imports
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from std_q7 import QualityTool

from collections import Counter


class Histogram(QualityTool):
    '''
    What is a histogram?
        -> a graph that represents the distribution of a quantitative variable.
    What use is a histogram?
        -> 
    '''
    def __init__(self, dataframe, feature):
        super().__init__()
        self.dataframe = dataframe
        self.feature = feature
        # Check the frame type of the data
        # must be of type list, array, series, ...
        # must be one-dimensional
        #
        # Check the type of the data
        # Handle currencies
        self.data = dataframe[feature]


    def clean_data(self):
        '''
        Removes:
        - currency symbols
        - empty strings
        - comma
        '''
        # Currency symbols
        try:
            for currency in ['€', '$', '£']:
                self.data = [element.replace(currency, '') for element in self.data]
        except:
            raise
        # Empty strings
        try:
            self.data = [element.replace(' ', '') for element in self.data]                        
        except:
            raise
        # Commas
        try:
            self.data = [float(element.replace(',', '.')) for element in self.data]                        
        except ValueError:
            print(
                'Feature \'{}\' cannot be converted into type \'float\''.format(
                feature))
            raise
        self.data = pd.Series(self.data)


    def outliers(self, quantile_sup, quantile_inf):
        '''
        Handles the outliers in the given data.
        '''
        if quantile_sup !=1 :
            temp_data = self.data[self.data < self.data.quantile(quantile_sup)]
        if quantile_inf != 0:
            temp_data = self.data[self.data > self.data.quantile(quantile_inf)]
        if quantile_inf == 0 and quantile_sup == 1:
            return self.data
        return temp_data


    def plot(self, quantile_sup=1, quantile_inf=0):
        # -------------
        # PRE-TREATMENT
        # -------------
        # Clean the data
        if type(self.data) == object:
            self.clean_data()
        # Handle the outliers
        temp_data = self.outliers(quantile_sup, quantile_inf)
        # -------------
        # MULTIPLE PLOT
        # -------------
        # First plot: the distribution
        fig = plt.figure(figsize=(10,5))
        plt.tight_layout(5.0)
        # plot the distribution
        ax1 = fig.add_subplot(121)
        plt.xticks(rotation=45,
                   ha='right')
        # Set xlim, ylim
        x_min = min(temp_data)
        x_max = max(temp_data)
        x_lim = (x_min*1.1, x_max*1.1)
        #
        # Set number of bins
        #n_bins = int(len(temp_data)/10)
        #
        # Set xlabel, ylabel
        plt.xlabel('Values')
        plt.ylabel('Count')
        # Set background style
        sns.set_theme(style="darkgrid")
        #
        # Set standard title
        plt.title('Distribution of ' + self.feature)
        
        plt.hist(temp_data,
                 edgecolor='k',
                 bins=30)
        # Second plot: the general information
        return super().general_info()


# TO DO
    def add_boxplot(self, ):
        pass