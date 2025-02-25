import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from prophet import Prophet

# Load Data
df = pd.read_csv('cleaned_netflix_data.csv')
df_c = pd.read_csv('Cleaned Data.csv')

# Load forecasts
forecast_total = pd.read_csv("forecast_total.csv")
forecast_movies = pd.read_csv("forecast_movies.csv")
forecast_tv = pd.read_csv("forecast_tv.csv")

# Convert ds column to DateTime
forecast_total['ds'] = pd.to_datetime(forecast_total['ds'])
forecast_movies['ds'] = pd.to_datetime(forecast_movies['ds'])
forecast_tv['ds'] = pd.to_datetime(forecast_tv['ds'])

#-------------------------Page Configuration------------------
st.set_page_config(page_title="NETFLIX TREND ANALYSIS", layout="centered")

# Netflix Theme
netflix_palette = ["#E50914", "#221F1F"]

#-------------------------Navigation Bar----------------------
tabs = st.tabs([
    'Overview',
    'Content Insights',
    'Trend Analysis',
    'Time Series Forecasting'
])

#---------------------TAB 1: Overview------------------------

with tabs[0]:
    st.markdown("<h1 style='text-align: center; font-size:30px;'>🎬 NETFLIX Data Analysis & Content Forecasting</h1>", unsafe_allow_html=True)

    st.markdown("""
        ---

        ### 🔍 **What You'll Find in This Dashboard**  
                

        1️⃣ **Netflix Content Release Forecast**
                
        📌 Predicts future Netflix content additions using time-series forecasting.  
    
        2️⃣ **Genre-wise Trends**
                
        📌 Tracks the rise and fall of different genres over the years.  
        📌 Identifies popular genres and evolving audience preferences.  

        3️⃣ **Movies vs. TV Shows Forecast**
                
        📌 Analyzes trends in movies vs. TV shows over time.  
        📌 Helps determine if Netflix is shifting towards more TV Shows or Movies.   

        ---

        ### 📊 **Visualizations Included**  

        ✅ **Pie Chart** – Movie Duration & TV Show Seasons Distribution  
        ✅ **Tree Chart** – Genre Distribution Across Countries  
        ✅ **Bar Chart** – Top 10 Genres  
        ✅ **Bar Chart** – Viewer Preferences for content from Certain Countries   
        ✅ **Bar Chart** – Top 10 Countries Producing Netflix Content  

        ---

        ### 📌 **Why This Matters?**  

        📢 **🎬 For Content Creators** – Identify trending genres to create relevant content.  
        📢 **📊 For Analysts & Researchers** – Explore content patterns & predict future trends.  
        📢 **👀 For Netflix Enthusiasts** – Discover how Netflix’s content library is evolving.  

        🎯 **Dive in and uncover Netflix insights like never before!** 🚀  
    """, unsafe_allow_html=True)


    

#---------------------TAB 2: Content Insights------------------
with tabs[1]:
    st.markdown('## CONTENT INSIGHTS')

    #----------------------------------------------------------------
    # 1. Pie Chart: Most Popular Type of Content (Movies vs. TV Shows)
    #----------------------------------------------------------------

    st.markdown('### 🔹Most Popular Type of Content (Movies vs. TV Shows)')

     # Count the number of Movies and TV Shows
    content_counts = df_c['Type'].value_counts()

    # Create Pie Chart
    fig_pie, ax_pie= plt.subplots()
    ax_pie.pie(content_counts, labels=content_counts.index, autopct='%1.1f%%', 
            colors=netflix_palette, startangle=90, wedgeprops={'edgecolor': 'black'})
    ax_pie.set_title("Proportion of Movies vs. TV Shows")

    # Display pie chart in Streamlit
    st.pyplot(fig_pie)

    #----------------------------------------------------------------
    # 2. Tree Chart: Genre Distribution Across Countries
    #----------------------------------------------------------------

    st.markdown('''
                
                ---

                ### 🔹Genre Distribution Across Countries''')

    # Filter out rows where Country or Genre is missing
    df_filtered = df_c.dropna(subset=['Country', 'Genre'])

    # Explode Genre (if multiple genres exist in one row)
    df_filtered['Genre'] = df_filtered['Genre'].str.split(', ')
    df_exploded = df_filtered.explode('Genre')

    # Count occurrences for better clarity
    df_grouped = df_exploded.groupby(['Country', 'Genre']).size().reset_index(name='Count')

    # Limit to Top 10 Countries for readability
    top_countries = df_grouped.groupby('Country')['Count'].sum().nlargest(10).index
    df_grouped = df_grouped[df_grouped['Country'].isin(top_countries)]

    # Create Treemap
    fig = px.treemap(df_grouped, path=['Country', 'Genre'], values='Count',
                    color='Genre', color_discrete_sequence=px.colors.qualitative.Prism)

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    #---------------------TAB 3: Trend Analysis------------------
with tabs[2]:
    st.markdown('## Trend Analysis')

    #----------------------------------------------------------------
    # 1. Bar Chart: Top 10 Genres
    #----------------------------------------------------------------
    st.markdown('### 🔹Top 10 Genres')

    df_c['main_genre'] = df_c['Genre'].str.split(',').str[0]
    genre_counts = df_c['main_genre'].value_counts().head(10)

    # Generate a color gradient from red
    fig_bar1, ax_bar1= plt.subplots()
    gradient_palette = sns.light_palette("#E50914", n_colors=len(genre_counts),reverse=True)
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette=gradient_palette)

    # Add count labels to each bar
    for i, v in enumerate(genre_counts.values):
        ax_bar1.text(v + 2, i, str(v), color='black', va='center', fontsize=12)  

    ax_bar1.set_xlabel("Count")
    ax_bar1.set_ylabel("Genre")
    ax_bar1.set_title("Top 10 Most Prevalent Genres")
    st.pyplot(fig_bar1)

        
    #----------------------------------------------------------------
    # 2. BAR Plot: Viewer Preferences for Content from Certain Countries
    #----------------------------------------------------------------

    st.markdown('''
                
                ---

                ### 🔹Viewer Preferences for Content from Certain Countries''')
    
    # Count movies & TV shows by country
    country_counts = df_c['Country'].value_counts().head(10)  # Top 10 countries
    
    # Plot
    fig_bar3, ax_bar3= plt.subplots()
    sns.barplot(x=country_counts.index, y=country_counts.values, palette=netflix_palette)

    ax_bar3.set_xticklabels(ax_bar3.get_xticklabels(), rotation=45, ha='right')
    ax_bar3.set_xlabel("Country")
    ax_bar3.set_ylabel("Number of Releases")
    ax_bar3.set_title("Top 10 Countries Producing Netflix Content")
    st.pyplot(fig_bar3) 

    #----------------------------------------------------------------
    # 3. Bar Chart: Top 10 Countries Producing Netflix Content 
    #----------------------------------------------------------------

    st.markdown('''
                
                ---
                
                ### 🔹Top 10 Countries Producing Netflix Content''')

    country_counts = df_c['Country'].value_counts().head(10)
    # Generate a color gradient from red
    fig_bar2, ax_bar2= plt.subplots()
    gradient_palette = sns.light_palette("#E50914", n_colors=len(genre_counts),reverse=True)
    sns.barplot(x=country_counts.values, y=country_counts.index, palette=gradient_palette)

    # Add count labels to each bar
    for i, v in enumerate(country_counts.values):
        ax_bar2.text(v + 2, i, str(v), color='black', va='center', fontsize=12)  

    ax_bar2.set_xlabel("Number of Releases")
    ax_bar2.set_ylabel("Country")
    ax_bar2.set_title("Top 10 Countries Producing Netflix Content")
    st.pyplot(fig_bar2)   

 
    #---------------------TAB 4: Time Series Forecasting------------------
with tabs[3]:
    st.markdown('### Time Series Forecasting')

    #----------------------------------------------------------------
    # 1.  Total Netflix Releases Forecast
    #----------------------------------------------------------------

    st.markdown("### 🔹 Total Netflix Releases Forecast")
    fig, ax = plt.subplots()
    ax.plot(forecast_total['ds'], forecast_total['yhat'], label="Total Releases", color='blue')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Releases")
    ax.legend()
    st.pyplot(fig)

    #----------------------------------------------------------------
    # 2. Genre-wise Trends
    #----------------------------------------------------------------

    # Genre-wise trends
    st.markdown('''
                
                ---
                
                ### 🔹Genre-wise Trends''')
    selected_genre = st.selectbox("Choose Genre:", df.columns[4:])
    fig, ax = plt.subplots()
    ax.plot(df['Year'], df[selected_genre], label=selected_genre, color='purple')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Releases")
    ax.legend()
    st.pyplot(fig)

    #----------------------------------------------------------------
    # 3. Movies vs. TV Shows Forecast
    #----------------------------------------------------------------

    st.markdown('''
                
                ---
                
                ### 🔹Movies vs. TV Shows Forecast''')
    fig, ax = plt.subplots()
    ax.plot(forecast_movies['ds'], forecast_movies['yhat'], label="Movies", color='red')
    ax.plot(forecast_tv['ds'], forecast_tv['yhat'], label="TV Shows", color='green')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Releases")
    ax.legend()
    st.pyplot(fig)











