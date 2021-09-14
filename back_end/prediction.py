# The present predict.py file is the intermediary between the dashboard and
# the fitted algorithm.
# - as inputs, he receives a row of features corresponding to the choosen
#   candidate.
# - as outputs, he returns to the dashboard the following elements:
#     * a judgement
#     * a probablity
#     * a plot explaining the relative position of the candidate among the 
#       population of the train set.



# --------------------
# IMPORTS
# --------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from xgboost import plot_importance



# --------------------
# CONSTANTS
# --------------------
CAT_COLS = ['CHANNEL_TYPE',
            'CODE_REJECT_REASON',
            'CREDIT_ACTIVE',
            'CREDIT_TYPE',
            'FONDKAPREMONT_MODE',
            'HOUSETYPE_MODE',
            'NAME_CASH_LOAN_PURPOSE',
            'NAME_CONTRACT_STATUS',
            'NAME_CONTRACT_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_GOODS_CATEGORY',
            'NAME_HOUSING_TYPE',
            'NAME_INCOME_TYPE',
            'NAME_PAYMENT_TYPE',
            'NAME_TYPE_SUITE',
            'OCCUPATION_TYPE', 
            'ORGANIZATION_TYPE',
            'WALLSMATERIAL_MODE',
            'WEEKDAY_APPR_PROCESS_START']


# --------------------
# GENERAL EVALUATION
# --------------------
def solvability_score(test_df, model, row):
    '''
    Return the probability in % for the sample to be attributed the note 1.
    model must be already fitted to the train set.
    '''
    # Solution de départ
    temp_df = test_df.copy()
    # Add the new row
    temp_df = temp_df.append(row)
    # Extract and post-process the score    
    app_predict_proba = model.predict_proba(temp_df)[:, 1][-1]
    app_predict_proba = round(app_predict_proba, 2) * 100
    return app_predict_proba


def credit_allocation(test_df, model, row):
    '''
    Provides with the color corresponding to the judgement.
    '''
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
    '''
    Determine if the applicant is eligible or not.
    '''
    temp_df = test_df.copy()
    temp_df = temp_df.append(row)
    app_predict = model.predict(temp_df)[-1]
    app_predict = int(app_predict)
    return app_predict
    

# --------------------
# STRENGTHS & WEAKNESSES
# --------------------
def most_important_features_table(model, X, n_feat=6):
    '''
    Display the n_feat most important feature.
    '''
    feature_importances = pd.Series(model.feature_importances_,
                                    index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    feature_importances = feature_importances[:n_feat]
    return feature_importances


def most_important_features_list(X, model, n_feat=6):
    '''
    Given a dataframe and a classification model, returns the list of the most
    important features.
    The model must be already trained on a train set.
    '''
    features = most_important_features_table(model, X, n_feat)
    features = list(features.index)
    return features


# --------------------
# PLOT THE CANDIDATE'S POSITION
# --------------------
def readable_string(my_string):
    '''
    Returns the name of the original qualitative feature without underscores.
    '''
    my_string = my_string.replace('_', ' ')
    my_string.capitalize()
    return my_string


def original_string(orig_col, option):
    '''

    '''
    new_string = orig_col.upper() + '_' + option
    new_string = new_string.replace(' ', '_')
    return new_string


def orig_encoded_feat(feature_name):
    '''
    Returns the name of the original qualitative feature.
    '''
    for cat_col in CAT_COLS:
        if cat_col in feature_name:
            return cat_col
    return False


def orig_encoded_option(feature_name):
    '''
    Returns the name of the original qualitative option.
    '''
    for cat_col in CAT_COLS:
        if cat_col in feature_name:
            return cat_col
    
    return False


def encoded_options(encoded_df, feature_name):
    '''
    Given an encoded column name, returns the list of the other options
    for the same original column before encoding.
    '''
    orig_col = orig_encoded_feat(feature_name)
    options_col = []
    if orig_col:
        for col in encoded_df.columns:
            if orig_col in col:
                options_col.append(col)
    return options_col


def app_spec_option(encoded_df, feature_name, app_id):
    '''
    Returns the name of the original qualitative option.
    '''
    options_col = encoded_options(encoded_df, feature_name)
    temp_dict = dict(encoded_df[options_col].loc[app_id])
    for key in temp_dict.keys():
        if temp_dict[key] == 1:
            option = key
    try:
        if option:
            return option
    except:
        return 'Not available'


def feature_type(test_df, model, feature_name):
    '''
    Determine the type of the feature:
    - Categorical,
    - Boolean,
    - or Qualitative
    '''
    if orig_encoded_feat(feature_name):
        return 'Qualitative'
    elif test_df[feature_name].nunique() == 2:
        return 'Boolean'
    else:
        return 'Quantitative'


def plot_customer_position(train_df, test_df, orig_train_df, model, row, feature_name):
    '''
    Plot the position of the applicant among the population of former
    applicants.
    '''
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
    '''
    Plot the position of the applicant for a boolean feature.
    Criterion is to have only 2 values in the column.
    '''
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
    '''
    Plot the position of the choosen customer among the population of other
    customers, for a quantitative feature.
    '''
    # POSITION OF THE APPLICANT
    app_feature_value = row[feature]
    app_feature_value = round(app_feature_value, 3)
    # Plot distributions
    h = sns.displot(train_df,
                    x=feature,
                    hue='TARGET',
                    kind='kde',
                    bw_adjust=.3,
                    fill=True)
    # Plot applicant position

    return h


def plot_cust_pos_qual_feat(train_df, test_df, orig_train_df, model, feature, row):
    '''
    Plot the position of the choosen customer among the population of other
    customers, for a quantitative feature.
    
    Warning: it is not very accurate to use orig_train_df, as it corresponds
    to the very original dataframe, without the feature engineering
    (nan values, outliers, ...).
    '''
    # Get the corresponding value for the customer
    value = row[feature]
    # Get the root feature
    original_col = orig_encoded_feat(feature)
    # Get the value taken by the candidate
    app_category = app_spec_option(test_df, feature, row['ID'])
    if value == 0:
        # Among columns coming from the same original_col,
        # find the one that contains 1
        for col in encoded_options(test_df, feature):
            if row[col] == 1:
                app_category = col
                app_category = app_category.split(original_col)[-1]
    else:
        app_category = feature.split(original_col)[-1]
    #
    try:
        app_category = app_category[1:].replace('_', ' ')
    except:
        # Data is not available, although the feature is important.
        # The applicant should give this information.
        return print('\'{}\' feature is not available, although'
                     ' important for the final decision.\n\nAsk the applicant'
                     ' for the corresponding information.'.format(
                         original_col))
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