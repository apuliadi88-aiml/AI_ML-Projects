import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load data
df = pd.read_csv('data/preprocessed_rapido_dataset.csv')

st.title("Rapido Ride-Sharing Data Dashboard")
st.markdown("""This dashboard provides insights into the ride-sharing data for Rapido. Explore various aspects of the data through the tabs below.""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pickup/Drop Heatmap",
    "Cancellations by Hour",
    "Surge Patterns",
    "Customer vs Driver Cancellations",
    "Cancellation rate by City and weekday"
])

# ----------------------
# Tab 1: Pickup/Drop Heatmap
# ----------------------
with tab1:
    st.header("Pickup/Drop City Heatmap")
    
    pivot = df.pivot_table(
        index='pickup_location',
        columns='drop_location',
        values='booking_id',
        aggfunc='count',
        fill_value=0
    )
    
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, cmap='Blues')
    plt.title('Number of Rides: Pickup vs Drop Location')
    st.pyplot(plt)

# ----------------------
# Tab 2: Cancellations by Hour
# ----------------------
with tab2:
    st.header("Cancellations by Hour")
    
    cancel_hour = df[df['booking_status']=='Cancelled'].groupby('hour_of_day').size().reset_index(name='cancel_count')
    
    fig = px.bar(cancel_hour, x='hour_of_day', y='cancel_count', 
                 labels={'hour_of_day':'Hour of Day', 'cancel_count':'Number of Cancellations'},
                 title='Cancellations by Hour')
    st.plotly_chart(fig)

# ----------------------
# Tab 3: Surge Patterns
# ----------------------
with tab3:
    st.header("Surge Multiplier Patterns")
    
    fig = px.histogram(df, x='avg_surge_multiplier', nbins=30,
                       title='Distribution of Surge Multiplier')
    st.plotly_chart(fig)
    
    fig = px.box(df, x='city', y='avg_surge_multiplier', points='all',
                 title='Surge Multiplier by City')
    st.plotly_chart(fig)

# ----------------------
# Tab 4: Customer Loyalty vs Driver Reliability score by City
# ----------------------
with tab4:
    st.header("Driver Reliability & Customer Loyalty by City")
    
    # Aggregate by city
    city_scores = df.groupby('city').agg({
        'driver_reliability_score':'mean',
        'customer_loyalty_score':'mean'
    }).reset_index()
    
    # Bar chart for Driver Reliability Score
    fig1 = px.bar(city_scores, x='city', y='driver_reliability_score',
                  title='Average Driver Reliability Score by City',
                  labels={'driver_reliability_score':'Driver Reliability Score', 'city':'City'})
    st.plotly_chart(fig1)
    
    # Bar chart for Customer Loyalty Score
    fig2 = px.bar(city_scores, x='city', y='customer_loyalty_score',
                  title='Average Customer Loyalty Score by City',
                  labels={'customer_loyalty_score':'Customer Loyalty Score', 'city':'City'})
    st.plotly_chart(fig2)
    
    # Optional: Combined scatter plot
    fig3 = px.scatter(city_scores, x='driver_reliability_score', y='customer_loyalty_score',
                      text='city', size='customer_loyalty_score', color='city',
                      title='Customer Loyalty vs Driver Reliability by City')
    st.plotly_chart(fig3)

# ----------------------
# Tab 5: Number of Cancellations by City and weekday heatmap
# ----------------------
with tab5:
    st.header("Number of Cancellations by City and Weekday")
    
    cancel_city_weekday = df.groupby(['city', 'day_of_week'])['booking_status'].apply(lambda x: (x=='Cancelled').sum()).reset_index(name='cancellation_count')
    
    pivot_cancel = cancel_city_weekday.pivot(index='city', columns='day_of_week', values='cancellation_count')
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot_cancel, annot=True, fmt='d', cmap='Reds')
    plt.title('Number of Cancellations by City and Weekday')
    st.pyplot(plt)