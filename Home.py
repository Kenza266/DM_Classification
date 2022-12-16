import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title('Dataset')
data = pd.read_csv('C:\\Users\\DELL\\Downloads\\M2Code\\DM\\Project\\DM-main\\Send\\pages\\datset.csv') 

st.dataframe(data)