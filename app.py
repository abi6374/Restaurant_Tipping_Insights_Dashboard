import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
@st.cache_data
def load_data():
    return sns.load_dataset('tips')

data = load_data()

# Title of the dashboard
st.title('Restaurant Tipping Insights Dashboard')

# Add description
st.write('This dashboard analyzes tipping patterns in restaurants based on various factors.')

# Create two columns for the first row
col1, col2 = st.columns(2)

# Box plot in first column
with col1:
    st.subheader('Tips by Sex')
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='sex', y='tip', data=data, ax=ax1)
    st.pyplot(fig1)
    plt.close(fig1)

# Scatter plot in second column
with col2:
    st.subheader('Total Bill vs Tip')
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='total_bill', y='tip', hue='time', data=data, ax=ax2)
    st.pyplot(fig2)
    plt.close(fig2)

# Heatmap in full width
st.subheader('Correlation Heatmap (Numeric Features)')
fig3, ax3 = plt.subplots(figsize=(10, 6))
# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(), annot=True, ax=ax3, cmap='coolwarm')
st.pyplot(fig3)
plt.close(fig3)