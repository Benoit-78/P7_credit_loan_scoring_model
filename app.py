
# Red to orange status applicant nb. 105966
# Orange to green status applicant: nb. 117298
# Green status applicant nb. 117417


# --------------------
# IMPORTS
# --------------------
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from back_end.prediction import *
from matplotlib import pyplot as plt, patches



# --------------------
# CONSTANTS
# --------------------
GITHUBPATH = 'https://github.com/Benoit-78/credit_loan_scoring_model'
#PATH = 'C:\\Users\\benoi\\Documents\\20.3 Informatique\\Data Science\\0_process\\P7 Modèle de scoring\\40 dossier_git'
PATH = GITHUBPATH
#os.chdir(PATH)
MODEL_PATH = PATH + '/blob/main/back_end/fitted_xgb.pkl?raw=true'

st.set_page_config(layout='centered')

st.write(PATH)
st.write(MODEL_PATH)

# --------------------
# LOAD DATA AND MODEL
# --------------------
#@st.cache(allow_output_mutation=True)
train_df, test_df, orig_train_df, model = load_data(PATH, MODEL_PATH)
# Get the most important features
main_features_row = most_important_features_list(test_df, model, n_feat=6)



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



# --------------------
# SETTINGS
# --------------------
st.sidebar.markdown('''---''')
st.sidebar.subheader("Settings")
# Reset values
#if st.sidebar.button('Update'):
#    row = orig_row
widgets = {}
for i, feature_name in enumerate(main_features_row):
    # Categorical variable
    if orig_encoded_feat(feature_name):
        # Label
        orig_col = orig_encoded_feat(feature_name)
        label = readable_string(orig_col)
        # Options
        options = list(orig_train_df[orig_col].dropna().unique())
        options.append('Not available')
        # Index
        app_option = app_spec_option(test_df, feature_name, row)
        for l, option in enumerate(options):
            if option in app_option:
                index = l
                break
        #st.write(' - '.join([label, option, str(index)]))
        widget_key = st.sidebar.selectbox(
            label=label,
            options=options,
            index=index)#,
            #on_change=refresh)
        # Save the original feature
        if widget_key != 'Not available':
            widget_key = original_string(orig_col, widget_key)
            widget_key = orig_encoded_option(widget_key)
            widgets[label] = widget_key
    # Boolean variable
    elif test_df[feature_name].nunique() == 2:
        if int(row[feature_name]) == 1:
            app_value = 1
        else:
            app_value = 0
        widgets[main_features_row[i]] = st.sidebar.radio(
            feature_name,
            (0, 1),
            index=app_value)
    # Quantitative variable
    else:
        min_value = int(min(test_df[feature_name]))
        max_value = int(max(test_df[feature_name]))
        app_value = int(test_df[feature_name].loc[applicant_id])
        widgets[main_features_row[i]] = st.sidebar.slider(feature_name.capitalize(), min_value, max_value, app_value)



# --------------------
# OUTPUTS
# --------------------
# Identify if the row is original or have been changed
new_row = row_from_widgets_dict(widgets.items(), row, test_df)
if not new_row.equals(row):    
    row = new_row



col1, col2 = st.columns([1, 4])
# First version of indicator and probability of reliability.
with col1:
    indic = decision_indicator(test_df, model)
    st.pyplot(indic.display_circle(row.drop('ID')))
with col2:
    scale = liability_scale(test_df, model)
    st.pyplot(scale.display_scale(row.drop('ID')))



# --------------------
# APPLICANT POSITION
# --------------------
st.markdown('''---''')
st.header("Applicant position")
col30, col31, col32 = st.columns(3)
with col30:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, main_features_row[0]))
with col31:    
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, main_features_row[1]))
with col32:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, main_features_row[2]))
col33, col34, col35 = st.columns(3)
with col33:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, main_features_row[3]))
with col34:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, main_features_row[4]))
with col35:
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, main_features_row[5]))



# --------------------
# OTHER CHARACTERISTICS
# --------------------
st.markdown('''---''')
st.header("Other characteristics")
orig_cols = []
for feature_name in test_df.columns:
    if orig_encoded_feat(feature_name):
        orig_cols.append(orig_encoded_feat(feature_name))
    else:
        orig_cols.append(feature_name)
orig_cols = list(dict.fromkeys(orig_cols))
non_modificable_cols = ['CNT FAM MEMBERS',
                        'EXT SOURCE 1',
                        'EXT SOURCE 2',
                        'EXT SOURCE 3',
                        'WEEKDAY APPR PROCESS START',
                        'HOUR APPR PROCESS START',
                        'log HOUR APPR PROCESS START prev']
for element in non_modificable_cols:
    orig_cols.remove(element)

col40, col41, col42 = st.columns([1, 3, 1])
with col41:
    feature = st.selectbox(label='',
                           options=orig_cols)
    st.pyplot(plot_customer_position(train_df, test_df, orig_train_df, model, row, feature))