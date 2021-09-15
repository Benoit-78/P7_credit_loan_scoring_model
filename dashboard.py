# Green status applicant nb. 105966

# --------------------
# IMPORTS
# --------------------
import matplotlib
import os
import pandas as pd
import streamlit as st
import xgboost as xgb
import streamlit.components.v1 as components

from back_end.prediction import *
from matplotlib import pyplot as plt, patches
from matplotlib import collections  as mc



# --------------------
# CONSTANTS
# --------------------
GITHUBPATH = 'https://github.com/Benoit-78/credit_loan_scoring_model'
PATH = 'C:\\Users\\benoi\\Documents\\20.3 Informatique\\Data Science\\0_process\\P7 Modèle de scoring\\40 dossier_git'
os.chdir(PATH)
MODEL_PATH = PATH + '\\back_end\\fitted_xgb.pkl'

st.set_page_config(layout='centered')

# --------------------
# LOADING
# --------------------
#@st.cache(allow_output_mutation=True)
def load_data(path, model_path):
    '''
    Make data and model ready to use.
    '''
    # Sample of processed TRAIN set
    train_df = pd.read_csv(PATH + '/data/app_samp_train.csv')
    train_df.set_index('SK_ID_CURR', inplace=True)
    # Sample of processed TEST set
    test_df = pd.read_csv(PATH + '/data/app_samp_test.csv')
    test_df.set_index('SK_ID_CURR', inplace=True)
    # Sample of unprocessed train set, to get the distributions
    orig_train_df = pd.read_csv(PATH + '/data/orig_train_samp.csv')
    orig_train_df.set_index('SK_ID_CURR', inplace=True)    
    for feature in orig_train_df.columns:
        orig_train_df[feature].replace('/', ' ', regex=True, inplace=True)
    # model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # Return 
    return train_df, test_df, orig_train_df, model

train_df, test_df, orig_train_df, model = load_data(PATH, MODEL_PATH)


for col in orig_train_df.columns[:10]:
    st.write(col)


# --------------------
# INPUTS
# --------------------
# Choose the applicant
st.sidebar.subheader('Selection')
applicant_id = st.sidebar.selectbox(
    'Applicant identification number',
    test_df.index)
orig_row = test_df.loc[applicant_id]
orig_row['ID'] = applicant_id
# Cleaning the orig_row from parasite characters
orig_row.rename(lambda x: x.replace('_', ' '), axis='index', inplace=True)
row = orig_row
# Reset the settings
back_to_original_row = st.sidebar.button('Update')
if back_to_original_row:
    row = orig_row


# --------------------
# SETTINGS
# --------------------
widgets = {}
placeholder = {}
st.sidebar.subheader("Settings")
# Get the most important features
IMPORTANT_FEATURES = most_important_features_list(test_df, model, n_feat=6)



j = -1 # row number
for i, feature_name in enumerate(IMPORTANT_FEATURES):
    # Categorical variable
    if orig_encoded_feat(feature_name):
        orig_col = orig_encoded_feat(feature_name)
        # Selectbox parameters
        label = readable_string(orig_col)
        options = list(orig_train_df[orig_col].dropna().unique())
        options.append('Not available')
        # Update the widget if necessary
        if back_to_original_row:
            index = int(orig_row[feature_name])
        else:
            for l, option in enumerate(options):
                if option.replace(' ', '') in feature_name.replace('_', ''):
                    index = l
        # Clean the options
        options = [option.replace('/', ' ').replace('_', ' ') for option in options]
        widget_key = st.sidebar.selectbox(
            label=label,
            options=options,
            index=index)
        # Save the original feature
        if widget_key != 'Not available':
            widget_key = original_string(orig_col, widget_key)
            widgets[feature_name] = 0
            widgets[widget_key] = 1
    # Boolean variable
    elif test_df[feature_name].nunique() == 2:
        if back_to_original_row:
            app_value = orig_row[feature_name]
        else:
            if int(test_df[feature_name].loc[applicant_id]) == 1:
                app_value = 1
            else:
                app_value = 0
        new_name = feature_name.replace('_', ' ').capitalize()
        widgets[feature_name] = st.sidebar.radio(
            new_name,
            (0, 1),
            index=app_value)
    # Quantitative variable
    else:
        min_value = int(min(test_df[feature_name]))
        max_value = int(max(test_df[feature_name]))
        app_value = int(test_df[feature_name].loc[applicant_id])
        widgets[feature_name] = st.sidebar.slider(feature_name.capitalize(), min_value, max_value, app_value)



if back_to_original_row:
    for key, value in widgets.items():
        widgets[key] = orig_row[key]



# --------------------
# OUTPUTS
# --------------------
# Credit status
class decision_indicator():
    '''
    Circle whose color can be red, orange or green according the decision taken.
    '''
    def __init__(self, test_df, model):
        self.df = orig_test_df
        self.model = model


    def display_circle(self, t_row):
        '''
        
        '''
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
    '''
    Object that displays the probability that the applicant to be reliable.
    '''
    def __init__(self, test_df, model):
        self.df = test_df
        self.model = model


    def display_scale(self, t_row):
        '''
        Displays the scale.
        '''
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



col1, col2 = st.columns([1, 4])
# First version of indicator and probability of reliability.
with col1:
    indic = decision_indicator(test_df, model)
    st.pyplot(indic.display_circle(row.drop('ID')))
with col2:
    scale = liability_scale(test_df, model)
    st.pyplot(scale.display_scale(row.drop('ID')))



# Identify if the row is original or have been changed
new_row = row.copy()
for key, value in widgets.items():
    clean_key = key.replace('_', ' ')
    new_row[clean_key] = value
if not new_row.equals(orig_row):
    row = new_row



# --------------------
# APPLICANT POSITION
# --------------------
st.write(test_df[IMPORTANT_FEATURES[1].replace('_', ' ')])
col30, col31, col32 = st.columns(3)
with col30:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, IMPORTANT_FEATURES[0].replace('_', ' ')))
with col31:    
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, IMPORTANT_FEATURES[1].replace('_', ' ')))
with col32:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, IMPORTANT_FEATURES[2].replace('_', ' ')))
col33, col34, col35 = st.columns(3)
with col33:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, IMPORTANT_FEATURES[3].replace('_', ' ')))
with col34:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, IMPORTANT_FEATURES[4].replace('_', ' ')))
with col35:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, IMPORTANT_FEATURES[5].replace('_', ' ')))



# Separation line
st.markdown('''---''')



# --------------------
# OTHER CHARACTERISTICS
# --------------------
st.header("Other characteristics")
orig_cols = []
for feature_name in test_df.columns:
    if orig_encoded_feat(feature_name):
        orig_cols.append(orig_encoded_feat(feature_name))
    else:
        orig_cols.append(feature_name)
orig_cols = list(dict.fromkeys(orig_cols))

non_modificable_cols = ['CNT_FAM_MEMBERS',
                        'EXT_SOURCE_1',
                        'EXT_SOURCE_2',
                        'EXT_SOURCE_3',
                        'WEEKDAY_APPR_PROCESS_START',
                        'HOUR_APPR_PROCESS_START',
                        'log_HOUR_APPR_PROCESS_START_prev']

for element in non_modificable_cols:
    orig_cols.remove(element)

col40, col41, col42 = st.columns([1, 2, 1])
with col41:
    feature = st.selectbox(label='',
                           options=orig_cols)
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, feature))