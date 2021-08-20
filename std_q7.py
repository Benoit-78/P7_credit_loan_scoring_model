# Author: Benoit DELORME
# Mail: delormebenoit211@gmail.com
# Creation date: 26/06/2021
# Main objective: provide an IT version of the tools of quality, as described
# by Dr. Ishikawa in its book 'Guide for Quality Control, 1968'


# imports
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter


class QualityTool():
    '''
    The quality tools are used in very different situations. 
    However, their purposes are often similar:
    - analysis,
    - control,
    - decision taking,
    - ...
    These common purposes are gathered in the present class.
    '''
    def __init__(self, purpose=None, scale=None, process=None, line=None,
        product=None, date=None, shift=None, multiple='Unique', divers=None):
        self.purpose = purpose
        self.scale = scale
        self.process = process
        self.line = line
        self.product = product
        self.date = date
        self.shift = shift
        self.divers = divers
        sns.set_theme(style="darkgrid")


    def general_info(self):
        temp_df = pd.DataFrame(columns=['Value'],
            index=['Purpose', 'Scale', 'Process', 'Line', 'Product', 'Date', 
            'Shift', 'Divers'],
            data=[self.purpose, self.scale, self.process, self.line,
            self.product, self.date, self.shift, self.divers])
        return temp_df  



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
        self.original_data = dataframe[feature]


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
                self.data = [element.replace(currency, '') for element in self.original_data]
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
        if quantile_sup:
            temp_data = self.data[self.data < self.data.quantile(quantile_sup)]
        if quantile_inf:
            temp_data = self.data[self.data > self.data.quantile(quantile_inf)]
        return temp_data


    def plot(self, quantile_sup=1, quantile_inf=0, n_rows=2):
        # -------------
        # PRE-TREATMENT
        # -------------
        # Clean the data
        self.clean_data()
        #
        # Handle the outliers
        temp_data = self.outliers(quantile_sup, quantile_inf)
        #
        # -------------
        # MULTIPLE PLOT
        # -------------
        # First plot: the distribution
        fig = plt.figure(figsize=(8,5))
        plt.tight_layout(5.0)
        # plot the distribution
        ax1 = fig.add_subplot(121)
        # Set xlim, ylim
        x_min = min(temp_data)
        x_max = max(temp_data)
        x_lim = (x_min*1.1, x_max*1.1)
        #
        # Set number of bins
        n_bins = int(len(temp_data)/10)
        #
        # Set xlabel, ylabel
        plt.xlabel('Values')
        plt.ylabel('Count')
        # Set background style
        sns.set_theme(style="darkgrid")
        #
        # Set standard title
        plt.title('Distribution of ' + self.feature)
        
        sns.histplot(temp_data,
                    bins=n_bins)
        # Second plt: the general information
        return super().general_info()



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
        temp_df = temp_df.dropna()
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
                x_inf = i
                y_inf = element
                x_sup = i+1
                y_sup = cum_frequencies[i+1]
                break
        # Abscissa of the 80% point (whose ordinate is by definition 80%).
        x_80 = (80-y_inf)*(x_sup-x_inf)/(y_sup-y_inf) + x_inf
        abs_80 = [x_80, 0]
        pt_80 = [x_80, 80]
        ord_80 = [len(frequencies)-1, 80]

        # Plot the graph
        factor = len(categories)
        fig = plt.figure(figsize=(math.sqrt(factor*5), math.sqrt(factor*3)))
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