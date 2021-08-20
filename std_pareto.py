# Author: Benoit DELORME
# Mail: delormebenoit211@gmail.com
# Creation date: 26/06/2021
# Main objective: provide an IT version of the Pareto diagram, as described
# by Dr. Ishikawa in its book 'Guide for Quality Control, 1968'


# imports
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from std_q7 import QualityTool

from collections import Counter


class Pareto(QualityTool):
    '''
    What is a Pareto graph?
    -> A bar graph of the most common elements of the list, sorted according
    their frequency
    What use is a Pareto graph?
    -> Identify the most common elements in a set and provide a basis for 
    prioritization of action.
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
        self.original_data = dataframe[feature]


    def plot(self, filter=None):
        '''
        Represent the repartition of data.
        '''
        sns.set_theme(style="darkgrid")
        # Extract list of categories and  their occurrence frequency
        temp_counter = Counter(self.dataframe[self.feature]).most_common()
        categories, frequencies = zip(*temp_counter)
        value_max = len(self.dataframe[self.feature])
        frequencies = [value/value_max*100 for value in frequencies]
        # Delete the NaN category
        temp_df = pd.DataFrame({'Categories':categories,
                                'Frequencies':frequencies})
        temp_df = temp_df.replace(np.nan, 'NaN')
        # Filter the elements
        temp_df.sort_values(by='Frequencies',
                            ascending=False,
                            inplace=True)
        if filter:
            categories = [element for element in temp_df['Categories']][:filter]
            frequencies= [element for element in temp_df['Frequencies']][:filter]
            cum_frequencies = list(np.array(frequencies).cumsum())
        else:
            categories = [element for element in temp_df['Categories']]
            frequencies= [element for element in temp_df['Frequencies']]
            cum_frequencies = list(np.array(frequencies).cumsum())
        # Determine 80% lines
        # Find the point on the cumulative sum curb that has a 80% ordinate
        for i, element in enumerate(cum_frequencies):
            if element > 80:
                eighty_percent = True
                x_inf = i
                y_inf = element
                x_sup = i+1
                y_sup = cum_frequencies[i+1]
                break
        # Abscissa of the 80% point (whose ordinate is by definition 80%).
        if eighty_percent:
            x_80 = x_inf + (80 - y_inf) / (y_sup - y_inf)
            abs_80 = [x_80, 0]
            pt_80 = [x_80, 80]
            ord_80 = [len(frequencies)-1, 80]
        # Plot the graph
        factor = len(categories)
        fig = plt.figure(figsize=(math.sqrt(factor*5),
                                  math.sqrt(factor*3)))
        # Bar graph
        ax1 = fig.add_subplot(111)
        plt.title('Pareto diagram, feature \'{}\''.format(self.feature))
        plt.xlabel('Categories')
        plt.xticks(rotation=45,
                   ha='right')
        plt.ylim((0, 110))
        plt.ylabel('Occurrence in % of total')
        ax1.bar(categories,
                frequencies,
                color='orange',
                edgecolor='k',
                alpha=0.5)
        # Cumulative sum graph
        ax2 = ax1.twinx()
        plt.ylabel('Cumulative sum')
        plt.ylim((0, 110))
        ax2.plot(range(0, len(cum_frequencies)),
                    cum_frequencies,
                    color='red')
        if eighty_percent:
            # Plot the 80% lines
            # Vertical line
            ax3 = ax2.twinx()
            plt.ylim((0, 110))
            ax3.plot([abs_80[0], pt_80[0]],
                    [abs_80[1], pt_80[1]],
                    '--',
                    color='red',
                    linewidth=1)
            # Horizontal line
            ax4 = ax2.twinx()
            plt.ylim((0, 110))
            ax4.plot([pt_80[0], ord_80[0]],
                    [pt_80[1], ord_80[1]],
                    '--',
                    color='red',
                    linewidth=1)
       	return super().general_info()