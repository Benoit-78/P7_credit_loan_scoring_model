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


class PieChart(QualityTool):
    '''
    What is a piechart?
        -> 
    What use is a piechart?
        -> 
    '''
    def __init__(self, df):
        super().__init__()
        self.df = df
        # Check the frame type of the data
        # must be of type list, array, series, ...
        # must be one-dimensional
        #
        # Check the type of the data
        # Handle currencies


    def filter(self, position, rate):
        '''
        Reject the values representing less than rate % of the given feature length.
        '''
        t_feat = list(self.df.columns)[position]
        t_dict = dict(Counter(self.df[t_feat]))
        t_len = self.df.shape[0]
        new_dict = {}
        for key, value in t_dict.items():
            if value/t_len > rate:
                new_dict[key] = value
        return new_dict


    def plot(self, position, rate=0.005):
        '''
        '''
        t_feat = list(self.df.columns)[position]
        # -------------
        # PRE-TREATMENT
        # -------------
        t_dict = self.filter(position, rate=rate)
        t_dict = {k: v for k, v in sorted(t_dict.items(), key=lambda item: item[1])}
        # -------------
        # MULTIPLE PLOT
        # -------------
        keys = list(t_dict.keys())
        values= list(t_dict.values())
        # First plot: the distribution
        fig, ax = plt.subplots(figsize=(8, 5),
                               subplot_kw=dict(aspect="equal"))
        ax.set_title('Unique elements of ' + t_feat + ' (>{}%)'.format(rate*100))
        # Set font size
        patches, texts, autotexts = ax.pie(values,
                                           autopct=lambda x: round(x, 1),
                                           startangle=90)
        ax.legend(patches,
                  keys,
                  title='Categories',
                  loc="best")
        plt.setp(autotexts, size=12, weight="bold")
        plt.show()
        # Second plot: the general information
        return super().general_info()