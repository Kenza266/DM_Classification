import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title('Dataset')
data = pd.read_csv('datset.csv') 

st.dataframe(data)
