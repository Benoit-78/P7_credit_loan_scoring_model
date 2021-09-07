# --------------------
# IMPORTS
# --------------------
import pandas as pd
import streamlit as st



# --------------------
# CONSTANTS
# --------------------
#IMPORTANT_FEATURES = 



@st.cache
def load_data(path):
    dataframe = pd.read_csv(path)
    return dataframe




# --------------------
# INPUTS
# --------------------
# Choose the applicant
applicant_id = st.sidebar.selectbox(
    'Applicant identification number',
    test_df.index)

# Get the 
add_selectbox = st.sidebar.slider('parametre_1', 0, 100, 25)

add_selectbox = st.sidebar.selectbox(
    'parametre_3',
    ('valeur_A', 'valeur_B', 'valeur_C'))



# --------------------
# OUTPUTS
# --------------------
col1, col2 = st.columns(2)
with col1:
    st.header("Credit status")
    with st.form("my_form_1"):
        st.subheader("rov Indicator")
        st.subheader("Decision")
        st.form_submit_button('')

with col2:
    st.header("Solvabilité")
    with st.form("my_form_2"):
        st.slider('Positionnement du candidat', 0, 100, 25)
        st.form_submit_button('')


col3, col4 = st.columns(2)
with col3:
    st.header('Recommandations')
    recommandations = pd.DataFrame(
        columns=['Indicateur', 'Recommandation'],
        index=['feature_1', 'feature_2', 'feature_3',
        'feature_4', 'feature_5', 'feature_6'])
    st.table(recommandations)

st.header('Points forts')

st.header('Points faibles')