import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from app import load_data

st.title('Trend Analysis')

data = load_data()

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(['Daily Patterns', 'Customer Behavior', 'Time Analysis'])

with tab1:
    st.subheader('Daily Patterns')
    
    # Average tips by day
    daily_avg = data.groupby('day')['tip'].agg(['mean', 'count']).reset_index()
    daily_avg.columns = ['Day', 'Average Tip', 'Number of Customers']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Average Tips by Day')
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=daily_avg, x='Day', y='Average Tip')
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader('Customer Traffic by Day')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=daily_avg, x='Day', y='Number of Customers')
        st.pyplot(fig2)
        plt.close(fig2)

with tab2:
    st.subheader('Customer Behavior Analysis')
    
    # Tips by party size
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='size', y='tip')
    plt.title('Tips by Party Size')
    st.pyplot(fig3)
    plt.close(fig3)
    
    # Average tip percentage by different factors
    data['tip_pct'] = (data['tip'] / data['total_bill']) * 100
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Tip % by Smoker/Non-smoker')
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=data, x='smoker', y='tip_pct')
        st.pyplot(fig4)
        plt.close(fig4)
    
    with col4:
        st.subheader('Tip % by Gender')
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=data, x='sex', y='tip_pct')
        st.pyplot(fig5)
        plt.close(fig5)

with tab3:
    st.subheader('Time Analysis')
    
    # Create pivot table for average tips by day and time
    pivot_data = pd.pivot_table(
        data,
        values='tip',
        index='day',
        columns='time',
        aggfunc='mean'
    )
    
    # Plot heatmap
    st.subheader('Average Tips by Day and Time')
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
    st.pyplot(fig6)
    plt.close(fig6)
    
    # Time distribution of customers
    st.subheader('Customer Distribution by Time')
    time_dist = data.groupby('time').size().reset_index()
    time_dist.columns = ['Time', 'Count']
    
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=time_dist, x='Time', y='Count')
    st.pyplot(fig7)
    plt.close(fig7)