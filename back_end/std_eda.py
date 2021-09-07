# Author: B.Delorme
# Mail: delormebenoit211@gmail.com
# Creation date: 23/06/2021
# Main objective: to provide a support for exploratory data analysis.

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import statistics as stat

import std_histogram
import std_pareto
import std_piechart

from collections import Counter


class Eda_Explorator():
    '''
    '''
    def __init__(self, dataset):
        '''
        '''
        self.dataset = dataset


    def readcsv(self, my_path, date_feature=[], dtype_dict={}, nan_values=[],
                true_values=[], false_values=[], nrows=500):
        '''
        Standardized csv file reading method.
        '''
        df = pd.read_csv(my_path,
                         parse_dates=date_feature,
                         dtype=dtype_dict,
                         na_value = nan_values,
                         true_values = true_values,
                         false_values= false_values,
                         nrows=nrows)
        return df


    def neat_int(self, t_int):
        '''

        '''
        return '{:,.0f}'.format(t_int)


    def neat_float(self, t_float):
        '''

        '''
        return '{:,.2f}'.format(t_float)


    def train_test_proportion(self, train_df, test_df):
        '''
        Plot the relative proportion of train and test set.
        '''
        plt.title('Train / test proportion')
        plt.pie(x=[train_df.shape[0], test_df.shape[0]],
                labels=['Train set', 'Test set'],
                autopct=lambda x: round(x, 1),
                startangle=90,
                wedgeprops={'edgecolor':'k', 'linewidth': 1})


    def nan_proportion(self, df):
        '''
        Returns the proportion of NaN values in the given dataframe.
        '''
        nan_proportion = df.isna().sum().sum() / df.size
        nan_proportion = int(nan_proportion*100)
        return nan_proportion


    def duplicates_proportion(self, df):
        '''
        Returns the proportion of duplicates values in the given dataframe.
        '''
        dupl_proportion = df.duplicated().sum() / df.shape[0]
        dupl_proportion = int(dupl_proportion*100)
        return dupl_proportion


    def dataset_infos(self):
        '''
        Returns the main caracteristics of the given dataframe.
        '''
        # Create the columns of the info dataframe
        info_df_columns = []
        info_df = pd.DataFrame(columns=['Rows', 'Features',
                                        'Size', 'Memory usage (bytes)',
                                        '% of NaN', '% of duplicates'],
                                index=[df.name for df in self.dataset])
        # Get the data
        row_list, feature_list, size_list, nan_list, mem_list, dupl_list = [],[],[],[],[],[]
        for i, df in enumerate(self.dataset):
            # Extract the data
            height = df.shape[0]
            width = df.shape[1]
            # Save the data
            row_list.append(height)
            feature_list.append(width)
            size_list.append(df.size)
            mem_list.append(df.memory_usage(deep=True).sum())
            nan_list.append(self.nan_proportion(df))
            dupl_list.append(self.duplicates_proportion(df))

        # Constitute the dataframe
        info_df['Rows'] = row_list
        info_df['Features'] = feature_list
        info_df['Size'] = size_list
        info_df['Memory usage (bytes)'] = mem_list
        info_df['% of NaN'] = nan_list
        info_df['% of duplicates'] = dupl_list
        # Compute the average values for each feature
        average_list = []
        for feat in info_df:
            average_list.append(stat.mean(info_df[feat]))
        info_df.loc['Average'] = average_list
        return info_df.astype(int)


    def dataset_plot(self):
        '''
        Plot the main caracteristics of each dataframe of the given dataset.
        Enable comparison.
        '''
        info_df = self.dataset_infos()
        return info_df.style.bar(color= 'lightblue', align='mid')


    def get_df_feat(self, df, descr_df, filter_feat, ext='csv'):
        '''
        Returns the feature descriptions of the given dataframe.
        - 'df' is the dataframe whose feature descriptions are wanted.
        - 'descr_df' is the dataframe that contains the feature descriptions.
        '''
        filtered_df = descr_df[descr_df[filter_feat]==df.name + '.' + ext]
        return filtered_df


    def is_two_categories(self, df, position, rate=0.99):
        '''
        Determine if the given feature contains two non-Boolean categories.
        '''
        # List of lists, to be enriched on the job.
        criteria_list = [('M', 'F'),
                         ('Male', 'Female')]
        # Get the list of unique elements
        t_feat = list(df.columns)[position]
        t_serie = df[t_feat].dropna()
        t_list = t_serie.unique()
        # Get the proportion of criteria list elements in the unique
        t_dict = dict(Counter(t_serie))
        # Determine if the sex names are present in the given feature
        indic_list = []
        for crit_list in criteria_list:
            if set(crit_list).intersection(set(t_list))==set(crit_list):
                indic_list.append(1)
                def_crit_list = crit_list
            else:
                indic_list.append(0)
        # Check if the proportion of sex names is relevant
        # 'rate' is an argument.
        if 1 in indic_list:
            weight = 0
            for sex in def_crit_list:
                weight += t_dict[sex]
            if weight/len(t_serie) > rate:
                return True
            else:
                return False
        else:
            return False


    def is_boolean(self, df, position):
        '''
        Determine if the feature is of boolean type or not.
        '''
        # Get the list of unique elements
        t_feat = list(df.columns)[position]
        t_serie = df[t_feat].dropna()
        t_list = t_serie.unique()
        if len(t_list) == 2:
            return True
        else:
            return False


    def is_there_a_big_group(self, df, position):
        '''
        Aim is to identify if there is a major group,
        overwhelming the other ones.
        '''


    def plot_feature(self, df, position,
                     quantile_sup=1, quantile_inf=0):
        '''
        Determine if the feature is qualitative or quantitative, then:
        - plot a Pareto diagram if qualitative;
        - plot a histogram if quantitative.
        '''
        t_feat = list(df.columns)[position]

        if self.is_boolean(df, position) or self.is_two_categories(df, position):
            my_piechart = std_piechart.PieChart(df)
            my_piechart.plot(position)
        elif df[t_feat].dtype != object:
            my_histogram = std_histogram.Histogram(df, t_feat)
            my_histogram.plot(quantile_sup=quantile_sup,
                              quantile_inf=quantile_inf)
        else:
            my_pareto = std_pareto.Pareto(df, t_feat)
            my_pareto.plot()


    def select_quant(self, df):
        '''
        Select only the quantitative features of the given dataframe.
        '''
        quant_df = pd.DataFrame()
        for col in list(df.columns):
            if df[col].dtype !=  object:
                quant_df[col] = df[col]
        return quant_df


    def quant_heatmap(self, df, drop_col=[], frac=1, annot=False):
        '''
        Display a heatmap of the correlation between features of a dataframe.
        '''
        # Drop given columns
        if len(drop_col) > 0:
            for col in drop_col:
                df = df.drop(col, axis=1)
        # Drop binary columns and two-categories features
        t_df = df
        for position in list(range(0, len(df.columns))):
            if self.is_boolean(df, position) or self.is_two_categories(df, position):
                t_feat = list(df.columns)[position]
                t_df = t_df.drop(t_feat, axis=1)
        # Select quantitative features only
        df = self.select_quant(t_df)
        # Cull a sample of the original dataframe
        df = df.sample(frac=frac)
        # Plot
        factor = len(df.columns)
        plt.figure(figsize=(factor*3,
                            factor*2/3))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr,
                                    dtype=bool))
        matrix = np.triu(corr)
        heatmap = sns.heatmap(corr,
                              mask=mask,
                              square=True,
                              linewidths=.2,
                              annot=annot,
                              center=0);
        heatmap.set_title('Correlation heatmap',
                          fontdict={'fontsize':15},
                          pad=12);


    def nan_proportion_features(self, df):
        '''
        Represent proportion of non-NaN data for each feature of the given dataframe.
        '''
        # Form the dataframe
        t_df = pd.DataFrame(columns=['keys', 'values', 'color'])
        for column in list(df.columns):
            if df[column].dtype != object:
                color = 'blue'
            else:
                color = 'orange'
            value = df[column].notna().sum()/df.shape[0]*100
            t_df.loc[t_df.shape[0]] = [column, value, color]
        # Sort by values
        t_df.sort_values(by='values', ascending=True, inplace=True)
        # Plot the graph
        plt.figure(figsize=(5, math.sqrt(5*df.shape[1])))
        plt.xlim((0, 105))
        plt.title('Proportion of non-NaN data' + ' ' + df.name)
        plt.barh(t_df['keys'],
                 t_df['values'],
                 color=t_df['color'],
                 alpha=0.5,
                 edgecolor='k')


    def df_min(self, df):
        '''
        Gives the minimum value of a dataframe
        '''
        return min(list(df.min()))


    def df_max(self, df):
        '''
        Gives the maximum value of a dataframe
        '''
        return max(list(df.max()))


    def qual_feat_corr(self, df, cols, rate=0.005):
        '''
        Returns a table of the correlations between categories of two qualitative
        series.
        Categories must represent at least (rate)% of the data.
        '''
        t_df = df[cols].dropna()
        df_len = df.shape[0]
        serie_1, serie_2 = t_df[cols[0]], t_df[cols[1]]
        counter_1, counter_2 = dict(Counter(serie_1)), dict(Counter(serie_2))
        list_1 = [key for key in counter_1.keys() if counter_1[key] > rate*df_len]
        list_2 = [key for key in counter_2.keys() if counter_2[key] > rate*df_len]

        corr_df = pd.DataFrame(columns=list_1, index=list_2)
        for key_1 in list_1:
            t_list = []
            for key_2 in list_2:
                temp_df = t_df[t_df[cols[0]]==key_1]
                temp_df = temp_df[t_df[cols[1]]==key_2]
                t_list.append(temp_df.shape[0])
            corr_df[key_1] = t_list
        
        if self.df_min(corr_df) < 0:
            v_min = self.df_min(corr_df)
        else:
            v_min =0
        if self.df_max(corr_df) > 0:
            v_max = self.df_max(corr_df)
        else:
            v_max = 0

        return corr_df.style.bar(color='lightblue',
                                 vmin=v_min,
                                 vmax=v_max)
        

    def cramers_v(self, serie_1, serie_2):
        """
        Calculate Cramers V statistic for categorial-categorial association.
        Uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        confusion_matrix = pd.crosstab(serie_1, serie_2)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


    def qual_df_corr(self, df, rate=0.005):
        '''
        Returns a heatmap of the correlations between qualitative features of the 
        given dataframe.
        '''
        # Get the qualitative features only
        qual_cols = []
        for col in list(df.columns):
            if df[col].dtype=='O':
                qual_cols.append(col)
        # Compute the coefficients dataframe
        corr_df = pd.DataFrame(columns=qual_cols,
                               index=qual_cols)
        list_2 = qual_cols.copy()
        for col_1 in qual_cols:
            corr_list = []
            for col_2 in qual_cols:
                if col_2 in list_2:
                    corr_list.append(
                        self.cramers_v(
                            df[col_1],
                            df[col_2]))
                else:
                    corr_list.append(np.nan)
            list_2.remove(col_1)
            corr_df[col_1] = corr_list
        # Formatting
        corr_df = corr_df.round(4)
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        matrix = np.triu(corr_df)
        # Return
        return sns.heatmap(corr_df,
                           mask=mask,
                           square=True,
                           linewidths=.2,
                           annot=False,
                           center=0.5)


    def optimize_floats(self, df):
        floats = df.select_dtypes(include=['float64']).columns.tolist()
        df[floats] = df[floats].apply(pd.to_numeric,
                                      downcast='float')
        return df


    def optimize_ints(self, df):
        ints = df.select_dtypes(include=['int64']).columns.tolist()
        df[ints] = df[ints].apply(pd.to_numeric,
                                  downcast='integer')
        return df


    def violinplot(df, cols,
               rate=0.005, quantile_sup=0.999, quantile_inf=0):
        '''
        Standard representation of correlation between a quantitative and a
        qualitative feature.
        '''
        #
        # Still to do: join the average points of all violins with a black line
        #
        col_qual = [col for col in cols if df[col].dtype == 'O'][0]
        col_quant = [col for col in cols if df[col].dtype != 'O'][0]
        # Filter qualitative
        t_dict = dict(Counter(df[col_qual]))
        t_len = df.shape[0]
        keys = []
        for key, value in t_dict.items():
            if value/t_len > rate:
                keys.append(key)
        df = df[df[col_qual].isin(keys)]
        # Filter quantitative feature
        df = df[df[col_quant] < quantile_sup*max(df[col_quant])]
        # Plot
        sns.violinplot(x=col_quant,
                       y=col_qual,
                       data=df[[col_quant, col_qual]])

# ----------------------------------------------------------------------------
# TO DO 
# ----------------------------------------------------------------------------
    def todo_df_plot(self):
        '''
        Return a family of plots representing the distributions of features
        of the given dataframe
        '''

    def todo_save_plot(self):
        '''
        '''


    def cat_bar_plot(self):
        '''
        Standard barh plot
        '''
        # To do:
        # 1) display the values and percentages on top of each bin
        # 2) color only one bin to enhance one category.



# ----------------------------------------------------------------------------
# ARCHIVE
# ----------------------------------------------------------------------------
    def old_dataset_plot(self):
        '''
        Plot the main caracteristics of each dataframe of the given dataset.
        Enable comparison.
        '''
        # Constitute the dataframe
        info_df = self.dataset_infos()
        # Plot settings
        wid_df = info_df.shape[1]
        n_cols = 4
        n_rows = wid_df//n_cols+1
        fig, axs = plt.subplots(ncols=n_cols,
                                nrows=n_rows,
                                figsize=(25, 5*n_rows),
                                sharey=True)
        fig.tight_layout()
        # Plot the graphs
        categories = list(info_df.index)
        for i, col in enumerate(info_df.columns):
            values = list(info_df[col])
            axs[i//n_cols, i%n_cols].set_title(col)
            if col == '% of NaN':
                axs[i//n_cols, i%n_cols].set_xlim((0, 100))
            axs[i//n_cols, i%n_cols].barh(categories,
                                         values,
                                         edgecolor='k',
                                         alpha=0.75)
        # Display blank space instead of empty subplots
        if wid_df%n_cols !=0:
            for i in list(range(wid_df%n_cols, n_cols)):
                axs[wid_df//n_cols, i].axis('off')


    def old_dataframe_infos(self, my_dataframe):
        '''
        Returns the main caracteristics of the given dataframe.
        '''
        # Create the columns of the info dataframe
        info_df = pd.DataFrame(columns=['Values'])
        # Extract the data
        height = my_dataframe.shape[0]
        width = my_dataframe.shape[1]
        # dtypes
        dtypes = [str(element) for element in list(my_dataframe.dtypes)]
        dtypes = dict(Counter(dtypes))
        # Memory usage
        mem_usage = my_dataframe.memory_usage(deep=True).sum()
        # Constitute the dataframe
        info_df.loc['Number of rows'] = height
        info_df.loc['Number of features'] = width
        for key, value in dtypes.items():
            info_df.loc[key] = value
        info_df.loc['Size'] = my_dataframe.size
        info_df.loc['NaN values proportion'] = self.nan_proportion(my_dataframe)
        info_df.loc['Memory usage (bytes)'] = mem_usage
        return info_df