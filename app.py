import streamlit as st
import pandas as pd

st.title("Vertical Performance Analysis Dashboard")

df = pd.read_csv("data/vertical_performance_results.csv")
st.dataframe(df)

health_counts = df["Health Zone"].value_counts()
st.bar_chart(health_counts)
