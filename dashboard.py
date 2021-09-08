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

st.set_page_config(layout="centered")

# --------------------
# LOADING
# --------------------
# data
#@st.cache#(allow_output_mutation=True)
def load_data(path):
    dataframe = pd.read_csv(path)
    return dataframe
train_df = load_data(PATH + '/data/app_samp_train.csv')
test_df = load_data(PATH + '/data/app_samp_test.csv')
orig_train_df = load_data(PATH + '/data/orig_train_samp.csv')
orig_test_df = load_data(PATH + '/data/orig_test.csv')

train_df.set_index('SK_ID_CURR', inplace=True)
test_df.set_index('SK_ID_CURR', inplace=True)
orig_train_df.set_index('SK_ID_CURR', inplace=True)
orig_test_df.set_index('SK_ID_CURR', inplace=True)

# model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)



# --------------------
# INPUTS
# --------------------
# Choose the applicant
st.sidebar.subheader('Applicant selection')
applicant_id = st.sidebar.selectbox(
    'Please select an identification number',
    test_df.index)
row = test_df.loc[applicant_id]


st.write(sample_judgement(test_df, model, row))


# Separation line
st.sidebar.markdown('''---''')


# Get the most important features
IMPORTANT_FEATURES = most_important_features_list(test_df, model, n_feat=6)
st.sidebar.subheader('Settings')
original_features = []
for i, feature_name in enumerate(IMPORTANT_FEATURES):
    if feature_qual(test_df, model, feature_name):
        orig_col = feature_qual(test_df, model, feature_name)
        st.sidebar.selectbox(
            orig_col.replace('_', ' ').capitalize(),
            orig_train_df[orig_col].dropna().unique())
        orig_col = orig_col.replace('_', ' ').capitalize()
        original_features.append(orig_col)
    elif test_df[feature_name].nunique() == 2:
        if int(test_df[feature_name].loc[applicant_id]) == 1:
            app_value = 1
        else:
            app_value = 0
        feature_name = feature_name.replace('_', ' ').capitalize()
        original_features.append(feature_name)
        st.sidebar.radio(
            feature_name,
            (0, 1),
            index=app_value)
    else:
        original_features.append(feature_name.capitalize())
        min_value = int(min(orig_test_df[feature_name]))
        max_value = int(max(orig_test_df[feature_name]))
        app_value = int(orig_test_df[feature_name].loc[applicant_id])
        st.sidebar.slider(feature_name.capitalize(), min_value, max_value, app_value)

# Separation line
st.sidebar.markdown('''---''')

# Ask for prediction
st.sidebar.button('Update')



# --------------------
# OUTPUTS
# --------------------
# Credit status
col1, col2 = st.columns([1,4])
with col1:
    st.header("Decision")
    color = credit_allocation(test_df, model, row)
    circle = matplotlib.patches.Circle(
        (0.5, 0.5),
        radius=0.2,
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(circle)
    ax.axis('off')
    st.pyplot(fig)
    # Text
    if color == 'red':
        judgement = 'Loan not granted'
    elif color == 'orange':
        judgement = 'Loan granted under conditions'
    elif color == 'green':
        judgement = 'Loan granted'
    st.write(judgement)



with col2:
    # Liability scale
    st.header("Score")
    lines = [[(0, 0), (45, 0)],
             [(45, 0), (50, 0)],
             [(50, 0), (100, 0)]]
    colors = np.array(['red', 'orange', 'green'])
    lc = mc.LineCollection(
        lines,
        colors=colors,
        linewidths=10,
        alpha=0.7)
    fig, ax = plt.subplots(figsize=(5, 1))
    # Applicant-specific arrow
    app_score = solvability_score(test_df, model, row)
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
    ax.text(45-3, 0.5, 45, fontsize=8)
    ax.text(50-1, 0.5, 50, fontsize=8)
    ax.text(100-2, 0.5, 100, fontsize=8)
    ax.axis('off')
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    st.write(fig)



st.markdown('''---''')



col5, col6 = st.columns(2)
with col5:
    st.header('Strengths')

with col6:
    st.header('Weaknesses')



st.markdown('''---''')



st.header('Recommandations')
# Display also the descriptions of the features, from description.csv
recommandations = pd.DataFrame(
    columns=['Indicator', 'Recommandation'],
    index=original_features)
st.table(recommandations)