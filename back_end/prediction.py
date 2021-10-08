# The present predict.py file is the intermediary between the dashboard and
# the fitted algorithm.
# - as inputs, he receives a row of features corresponding to the choosen
#   candidate.
# - as outputs, he returns to the &ard the following elements:
#     * a judgement
#     * a probablity
#     * a plot explaining the relative position of the candidate among the 
#       population of the train set.



# IMPORTS
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import seaborn as sns
import streamlit as st
import urllib.request
import xgboost as xgb

from matplotlib import collections  as mc
from xgboost import plot_importance
from matplotlib import pyplot as plt, patches



# CONSTANTS
CAT_COLS = ['CHANNEL TYPE',
            'CODE REJECT REASON',
            #'CREDIT_ACTIVE',
            'CREDIT TYPE',
            'FONDKAPREMONT MODE',
            'HOUSETYPE MODE',
            #'NAME_CASH_LOAN_PURPOSE',
            'NAME CONTRACT STATUS',
            'NAME CONTRACT TYPE',
            'NAME EDUCATION TYPE',
            'NAME FAMILY STATUS',
            'NAME GOODS CATEGORY',
            'NAME HOUSING TYPE',
            'NAME INCOME TYPE',
            'NAME PAYMENT TYPE',
            'NAME TYPE SUITE',
            'OCCUPATION TYPE', 
            'ORGANIZATION TYPE',
            'WALLSMATERIAL MODE',
            'WEEKDAY APPR PROCESS START']



# PROCESS AND CLEANING
def load_data(path):
    """Load .csv data and makes them easy to use."""
    # Sample of processed TRAIN set
    #train_df = pd.read_csv(path + '\\data\\app_samp_train.csv')
    train_df = pd.read_csv(path + '/blob/main/data/app_samp_train.csv?raw=true')
    train_df.set_index('SK_ID_CURR', inplace=True)
    # Sample of processed TEST set
    #test_df = pd.read_csv(path + '\\data\\app_samp_test.csv')
    test_df = pd.read_csv(path + '/blob/main/data/app_samp_test.csv?raw=true')
    test_df.set_index('SK_ID_CURR', inplace=True)
    # Sample of unprocessed train set, to get the distributions
    #orig_train_df = pd.read_csv(path + '\\data\\orig_train_samp.csv')
    orig_train_df = pd.read_csv(path + '/blob/main/data/orig_train_samp.csv?raw=true')
    orig_train_df.set_index('SK ID CURR', inplace=True)    
    for feature in orig_train_df.columns:
        orig_train_df[feature].replace('/', ' ', regex=True, inplace=True)
        orig_train_df[feature].replace('_', ' ', regex=True, inplace=True)
    # Return 
    return train_df, test_df, orig_train_df


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = 'https://drive.google.com/drive/folders/1ZB4fXNTjHRDpyePpdCCdqmihF0s-svAB?usp=sharing'
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)


def load_model(model_path):
    """Load pickelised model"""
    # model
    #model = xgb.XGBClassifier()
    #model.load_model(model_path)
    #model = pickle.load(open(model_path, 'rb'))
    model = pickle.load(urllib.request.urlopen(model_path))
    return model


def readable_string(my_string):
    """Returns the name of the original qualitative feature without underscores."""
    my_string = my_string.replace('_', ' ')
    my_string.capitalize()
    return my_string


def original_string(orig_col, option):
    """

    """
    new_string = orig_col.upper() + ' ' + option
    #new_string = new_string.replace(' ', '_')
    return new_string


def orig_encoded_feat(feature):
    """Returns the name of the original qualitative feature."""
    for cat_col in CAT_COLS:
        if cat_col in feature:
            return cat_col
    return False


def orig_encoded_option(feature):
    """
    Returns the name of the original qualitative option.
    """
    for cat_col in CAT_COLS:
        if cat_col in feature:
            return feature.split(cat_col)[-1]
    return False


def encoded_options(encoded_df, feature_name):
    """
    Given an encoded column name, returns the list of the other options
    for the same original column before encoding.
    """
    orig_col = orig_encoded_feat(feature_name)
    options_col = []
    if orig_col:
        for col in encoded_df.columns:
            if orig_col in col:
                options_col.append(col)
    return options_col


def app_spec_option(df, feature, row):
    """
    Returns the name of the qualitative option of the given row.
    """
    
    if int(row[feature]) == 1:
        option = orig_encoded_option(feature)
    elif int(row[feature]) == 0:
        orig_col = orig_encoded_feat(feature)
        options_cols = [col for col in df.columns if orig_col in col]
        for col in options_cols:
            if row[col] == 1:
                option = orig_encoded_option(col)
        try:
            option
        except:
            option = 'Not available'
    return option


def feature_type(test_df, model, feature_name):
    """
    Determine the type of the feature:
    - Categorical,
    - Boolean,
    - or Qualitative
    """
    if orig_encoded_feat(feature_name):
        return 'Qualitative'
    elif test_df[feature_name].nunique() == 2:
        return 'Boolean'
    else:
        return 'Quantitative'


def clean_df(df):
    """

    """
    return df.rename(lambda x: x.replace('_', ' '), axis='columns')


def sets_difference(df, row):
    """

    """
    row_set = set(row.index)
    df_set = set(df.columns)
    return df_set.difference(row_set), row_set.difference(df_set)


def row_from_widgets_dict(widgets_dict, orig_row, encoded_df):
    """
    Make a row out of the widgets results.
    Must take into account the updates.
    """
    new_row = orig_row.copy()
    for key, value in widgets_dict:
        # Qualitative values
        if key in CAT_COLS:
            option_cols = encoded_options(encoded_df, key+value)
            # Reset to 0 the other options
            for option_col in option_cols:
                new_row[option_col] = 0
            # Set to 1 the new option choosen
            new_row[key+value] = 1
        # Quantitative and boolean values
        else:
            new_row[key] = value
    return new_row


# MOST IMPORTANT FEATURES
def most_important_features_table(X, model, n_feat=6):
    """
    Display the n_feat most important feature.
    """
    feature_importances = pd.Series(model.feature_importances_,
                                    index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    feature_importances = feature_importances[:n_feat]
    return feature_importances


def most_important_features_list(X, model, n_feat=6):
    """
    Given a dataframe and a classification model, returns the list of the most
    important features.
    The model must be already trained on a train set.
    """
    features = most_important_features_table(X, model, n_feat)
    features = list(features.index)
    return features


# APPLICANT STATUS
def solvability_score(test_df, model, row):
    """
    Return the probability in % for the sample to be attributed the note 1.
    model must be already fitted to the train set.
    """
    temp_df = test_df.copy()
    # Adapt the columns names, without underscore
    temp_df = clean_df(temp_df)
    # Add the new row
    temp_df = temp_df.append(row)
    # Extract the score    
    app_predict_proba = model.predict_proba(temp_df)
    app_predict_proba = app_predict_proba[:, 1][-1]
    # Post-process
    app_predict_proba = round(app_predict_proba, 2) * 100
    return app_predict_proba


def credit_allocation(test_df, model, row):
    """
    Provides with the color corresponding to the judgement.
    """
    score = solvability_score(test_df, model, row)
    if score < 40 and score >= 0:
        note = 'red'
    elif score < 50:
        note = 'orange'
    elif score <= 100:
        note = 'green'
    else:
        return 'Error in score evaluation. See predict.py script.'
    return note


def sample_judgement(test_df, model, row):
    """
    Determine if the applicant is eligible or not.
    """
    temp_df = test_df.copy()
    temp_df = clean_df(temp_df)
    return temp_df.shape, len(row)
    temp_df = temp_df.append(row)
    app_predict = model.predict(temp_df)[-1]
    app_predict = int(app_predict)
    return app_predict
    

# PLOT THE APPLICANT'S POSITION
class decision_indicator():
    """
    Circle whose color can be red, orange or green according the decision taken.
    """
    def __init__(self, test_df, model):
        self.df = test_df
        self.model = model


    def display_circle(self, t_row):
        """

        """
        st.header("Decision")
        color = credit_allocation(self.df, self.model, t_row)
        # Colored circle
        circle = matplotlib.patches.Circle(
            (0.5, 0.5),
            radius=0.2,
            color=color,
            alpha=0.6,
            edgecolor="black",
            linewidth=1)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_patch(circle)
        ax.axis('off')
        return fig



class liability_scale():
    """
    Object that displays the probability that the applicant to be reliable.
    """
    def __init__(self, test_df, model):
        self.df = test_df
        self.model = model


    def display_scale(self, t_row):
        """
        Displays the scale.
        """
        st.header("Score")
        lines = [[(0, 0), (40, 0)],
                 [(40, 0), (50, 0)],
                 [(50, 0), (100, 0)]]
        colors = np.array(['red', 'orange', 'green'])
        lc = mc.LineCollection(
            lines,
            colors=colors,
            linewidths=10,
            alpha=0.6)
        fig, ax = plt.subplots(figsize=(5, 1))
        # Applicant-specific arrow
        app_score = solvability_score(self.df, self.model, t_row)
        ax.arrow(app_score, -2,
                 0, 1,
                 head_width=2, head_length=0.5,
                 fc='k', ec='k')
        # Text with applicant score
        ax.text(app_score-2, -2.5,
                int(app_score),
                fontsize=8)
        # Text of scale
        ax.text(0-1, 0.5, 0, fontsize=8)
        ax.text(10-1, 0.5, 'Not reliable', fontsize=8, color='r')
        ax.text(40-3, 0.5, 40, fontsize=8)
        ax.text(50-1, 0.5, 50, fontsize=8)
        ax.text(68-1, 0.5, 'Reliable', fontsize=8, color='g')
        ax.text(100-2, 0.5, 100, fontsize=8)
        # Other settings
        ax.axis('off')
        # Plot
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        return fig



def plot_customer_position(train_df, test_df, orig_train_df, model, row, feature_name):
    """
    Plot the position of the applicant among the population of former
    applicants.
    """
    featuretype = feature_type(test_df, model, feature_name)
    if featuretype == 'Qualitative':
        #return 'Categorical'
        return plot_cust_pos_qual_feat(train_df,
                                       test_df,
                                       orig_train_df,
                                       model,
                                       feature_name,
                                       row)
    elif featuretype == 'Boolean':
        #return 'Boolean'
        return plot_cust_pos_bool_feat(train_df,
                                       test_df,
                                       model,
                                       feature_name,
                                       row.drop('ID'))
    else:
        #return 'Quantitative'
        return plot_cust_pos_quant_feat(train_df,
                                        test_df,
                                        feature_name,
                                        row.drop('ID'))


def plot_cust_pos_bool_feat(train_df, test_df, model, feat, row):
    """
    Plot the position of the applicant for a boolean feature.
    Criterion is to have only 2 values in the column.
    """
    value = row[feat]
    judgement = sample_judgement(test_df, model, row)
    # Get the distribution with/without the feature, granted/refused loan.
    without_df = train_df[train_df[feat]==0]
    with_df = train_df[train_df[feat]==1]
    without_granted = without_df[without_df['TARGET']==1].shape[0]
    with_granted = with_df[with_df['TARGET']==1].shape[0]
    if value == 0:
        colors = ['blue', 'lightgrey']
    else:
        colors = ['lightgrey', 'blue']
    fig, ax = plt.subplots()
    ax.set_ylabel('Chance of success',
                  fontsize=20)
    ax.set_title(feat, fontsize=30)
    ax.bar(['Without', 'With'],
           [without_granted, with_granted],
           color=colors,
           edgecolor='k',
           alpha=0.5,
           linewidth=1)
    return fig
    

def plot_cust_pos_quant_feat(train_df, test_df, feature, row):
    """
    Plot the position of the choosen customer among the population of other
    customers, for a quantitative feature.
    """
    # Applicant's position
    app_value = row[feature]
    app_value = round(app_value, 3)
    # Plot
    distr = train_df[train_df['TARGET']==1]
    fig, (ax_box, ax_dist) = plt.subplots(2, 1,
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [1, 5]})
    fig.suptitle('Repartition of successful applicants')
    # Show main characteristics
    sns.boxplot(distr[feature],
                color='green',
                boxprops=dict(edgecolor='k',
                              alpha=0.5),
                ax=ax_box)
    ax_box.set(xlabel='')
    # Plot applicant position
    ax_dist.axvline(x=app_value,
                    c='k',
                    linestyle='--',
                    label='Selected applicant',
                    linewidth=1)
    ax_dist.legend(loc='best')
    # Show distribution
    sns.kdeplot(data=distr,
                 x=feature,
                 color='green',
                 alpha=0.5,
                 bw_adjust=.3,
                 fill=True,
                 ax=ax_dist)

    return fig


def plot_cust_pos_qual_feat(train_df, test_df, orig_train_df, model, feature, row):
    """
    Plot the position of the choosen customer among the population of other
    customers, for a quantitative feature.

    Warning: it is not very accurate to use orig_train_df, as it corresponds
    to the very original dataframe, without the feature engineering
    (nan values, outliers, ...).
    """
    # Get the corresponding value for the customer
    value = row[feature]
    # Get the root feature
    original_col = orig_encoded_feat(feature)
    # Get the value taken by the candidate
    app_category = app_spec_option(test_df, feature, row)
    if value == 0:
        # Among columns coming from the same original_col,
        # find the one that contains 1
        for col in encoded_options(test_df, feature):
            if row[col] == 1:
                app_category = col
                app_category = app_category.split(original_col)[-1]
    else:
        app_category = feature.split(original_col)[-1]
    app_category = app_category[1:]
    # Get the categorical distribution for accepted applicants
    train_1_df = orig_train_df[orig_train_df['TARGET']==1]
    train_1_df = pd.DataFrame(train_1_df[original_col].value_counts())
    # Get the categorical distribution for refused applicants
    train_0_df = orig_train_df[orig_train_df['TARGET']==0]
    train_0_df = pd.DataFrame(train_0_df[original_col].value_counts())
    # Merge the two dataframe
    pareto_df = pd.merge(train_1_df,
                         train_0_df,
                         how='outer',
                         left_index=True,
                         right_index=True,
                         suffixes=['_accepted', '_refused'])
    pareto_df.replace(np.nan, 0, inplace=True)
    # Create a column sum
    pareto_df['Ratio'] = pareto_df[pareto_df.columns[1]] / pareto_df[pareto_df.columns[0]]
    pareto_df.sort_values(by='Ratio', ascending=False, inplace=True)
    # Clean the data
    clean_index = [index.replace('/', '_') for index in pareto_df.index]
    pareto_df.reset_index(drop=True, inplace=True)
    pareto_df['new_index'] = clean_index
    pareto_df.set_index('new_index', inplace=True)
    # Create the figure
    fig, ax = plt.subplots()
    # Plot parameters
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right',
             fontsize=15)
    ax.set_ylabel('Chance of success', fontsize=20)
    # Color the semi-column of the customer choosen.
    colors=[]
    for category in pareto_df.index:
        if category == app_category:
            colors.append('blue')
        else:
            colors.append('lightgrey')
    # Plot the bar graph
    ax.set_title(original_col, fontsize=30)
    ax.bar(pareto_df.index,
            pareto_df['Ratio'],
            color=colors,
            alpha=0.5,
            edgecolor='k')
    # Other parameters
    return fig