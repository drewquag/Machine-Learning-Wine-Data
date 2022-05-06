import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

#Read in the datafram from a zipfile
zf = zipfile.ZipFile('winemag-data_first150k.zip')

df = pd.read_csv(zf.open('winemag-data_first150k.csv'), index_col=0)


#Create a function that returns a dataframe filtered by country, province (optional), and/or region (optional)
def display_by_region (country_province_region):
    if len(country_province_region) == 1:
        df_country = df[df['country'] == country_province_region[0]]
        return df_country
    elif len(country_province_region) == 2:
        df_country = df[df['country'] == country_province_region[0]]
        df_province = df_country[df_country['province'] == country_province_region[1]]
        return df_province
    elif len(country_province_region) == 3:
        df_country = df[df['country'] == country_province_region[0]]
        df_province = df_country[df_country['province'] == country_province_region[1]]
        df_region = df_province[df_province['region_1'] == country_province_region[2]]
        return df_region

#Create a function that creates a word cloud based on the descriptions of wine found in a dataframe
def word_cloud(df):
    text = " ".join(review for review in df['description'])

    wordcloud = WordCloud().generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

#Create a pie chart that maps the percentage of each wine variety for a country
def find_wine_variety_frequency(country):
    df_country = df[df['country'] == country]
    df_variety = df_country.groupby('variety').size().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    plt.title(f'Frequency of Top Wine Varieties in {country}')
    ax.pie(df_variety, autopct='%1.0f%%', labels=df_variety.index)
    return fig


# Create Streamlit App

option = st.sidebar.selectbox('Please choose one of the following pages to display', ['Home Page', 'Word Cloud', 'Machine Learning','Wine Quality by Country' ])
st.sidebar.write('*' * 100)

st.title("Andrew Quagliaroli's Data Science Final")
st.write('*' * 100)

#Design homepage
if option == 'Home Page':
    col1, col2= st.columns(2)
    with col2:
        st.image('IMG_4272.jpg', width=250)
    with col1:
        st.write('**This is my data science project which displays various insights into a dataset containing over 150k wine descriptions.**')
        st.markdown('**To begin, please navigate the sidebar to view various web pages.**')
        st.markdown('_Below I have included both the link to the code which created this entire project and a project report summarizing key findings:_')
        st.write("-  check out this [Deepnote Code](https://deepnote.com/workspace/drewquag-2e9b6724-412b-4149-af9a-d088e32ec141/project/Final-Project-71878eba-a214-42c9-ad82-39b82a172ad5/%2FFinal%20Python%20Work.ipynb)")
        st.write("-  check out this [Project Report](https://bentleyedu-my.sharepoint.com/:w:/g/personal/aquagliaroli_falcon_bentley_edu/ERsa1G2R641LvZQrWrqZx1IB3VS09oqkKpIesEoLYt9yqg?e=itv3b3)")


#Create word cloud page with option to select the geographical region for the wordcloud
elif option == 'Word Cloud':
    st.header('Wordcloud Vizualization')
    st.write('On this page you are able to vizualize a wordcloud of the most used words in the descriptions of wine based on country, province, or region. Please choose an option on the sidebar to begin vizualizing.')
    st.write('-' * 100)
    scale = st.sidebar.selectbox('Would you like to view a word cloud by country, Province, or region?', ['<Select Option>', 'Country', 'Province', 'Region'])
    st.sidebar.write('*' * 100)
    if scale == 'Country':
        country = st.sidebar.selectbox('Please choose a country', df['country'].dropna().sort_values().unique())
        st.subheader(f'Word Cloud of Wine Descriptions: {country}')
        df_wordcloud = display_by_region([country])
        word_cloud(df_wordcloud)
    elif scale == 'Province':
        country = st.sidebar.selectbox('Please choose a country', df['country'].dropna().sort_values().unique())
        if len(display_by_region([country])['province'].dropna().sort_values().unique()) == 0:
            st.write('**_This country does not have a province, please either choose a higher geographical region or choose another combination of locations._**')
        else:
            province = st.sidebar.selectbox('Please choose a Province within your selected country', display_by_region([country])['province'].dropna().sort_values().unique())
            st.subheader(f'Word Cloud of Wine Descriptions: {province}, {country}')
            df_wordcloud = display_by_region([country, province])
            word_cloud(df_wordcloud)

    elif scale == 'Region':
        country = st.sidebar.selectbox('Please choose a country', df['country'].dropna().sort_values().unique())
        province = st.sidebar.selectbox('Please choose a Province within your selected country', display_by_region([country])['province'].dropna().sort_values().unique())
        if len(display_by_region([country, province])['region_1'].dropna().sort_values().unique()) == 0:
            st.write('**_This province does not have a subregion, please either choose a higher geographical region or choose another combination of locations._**')
        else:
            region = st.sidebar.selectbox('Please choose a region within your selected Province', display_by_region([country, province])['region_1'].dropna().sort_values().unique())
            st.subheader(f'Word Cloud of Wine Descriptions: {region}, {province}, {country}')
            df_wordcloud = display_by_region([country, province, region])
            word_cloud(df_wordcloud)
            
#Create wine quality vizualization page with option to pick country
elif option == 'Wine Quality by Country':
    st.header('Frequencies of Wine type by country')
    st.write('_This page will allow you to view the best wines varieties made in each country (max 10 wine varieties). This is done by averaging the points for each wine variety in each respective country and then ranking them._')
    country = st.sidebar.selectbox('Please choose a country', df['country'].dropna().sort_values().unique())
    st.pyplot(find_wine_variety_frequency(country))
    st.write('_Here is a list of the top wineries (max 10) in your chosen country ranked by average points received by every wine they produce._')
    top_10 = display_by_region([country]).groupby('winery')['points'].mean().sort_values(ascending=False).head(10)
    st.subheader(f'Top Wineries with Highest Average Wine Ratings in {country}')
    st.write(top_10)

#Create Machine Learning Model Page
elif option == 'Machine Learning':
    features = st.sidebar.button('Visualize Features')
    model = st.sidebar.button('Machine Learning Performance')
    st.header('Machine Learning with Wine Descriptions')
    if features:
        range_ = st.sidebar.slider('Please choose a range of wine ratings you would like to discover:', int(df['points'].min()), int(df['points'].max()), (int(df['points'].min()), int(df['points'].max())))
        df = df.query(f"points >= {range_[0]} and points <= {range_[1]}")
        st.write('_On this page you will be able to visualize some of the key features that make our predictive model possible.\nWe will be training the descriptions of each wine to predict the points that it will be scored.\nFeel free to use the slider on the sidebar to look at a specific subset of ratings._')

        # Make bar chart
        st.subheader('_Number of Wines at Each Point Rating_')
        val_count = df['points'].value_counts()
        fig, ax = plt.subplots(figsize=(30,10))
        sns.barplot(val_count.index, val_count.values, alpha=0.8)
        ax.set_title('Number of wines per point', fontweight="bold", size=25)
        plt.xticks(fontsize=20) # X Ticks
        plt.yticks(fontsize=20) # Y Ticks
        ax.set_ylabel('Number of wines', fontsize = 25)
        ax.set_xlabel('Points', fontsize = 25)
        st.write('As you can see, the distribution of wines across points is approximately normal.')

        st.pyplot(fig)

        # Make box and whisker
        st.subheader('_Box & Whisker Plot for Length of Wine Descriptions_')
        st.write('As you can see, wines with a higher point rating tend to have longer descriptions.')

        fig, ax = plt.subplots(figsize=(30,10))

        df = df.assign(description_length = df['description'].apply(len))

        sns.boxplot(x='points', y='description_length', data=df)

        plt.xticks(fontsize=20) # X Ticks
        plt.yticks(fontsize=20) # Y Ticks
        ax.set_title('Description Length per Points', fontweight="bold", size=25)

        ax.set_ylabel('Description Length', fontsize = 25)
        ax.set_xlabel('Points', fontsize = 25)

        st.pyplot(fig)
    elif model:
        st.subheader('Machine Learning Model Performance Metrics')
        st.write('This table below gives both the precision and the recall of our machine learning model. The precision measures how many false negatives our model created and the recall measures the number of false negatives of our model.')
        st.write('This then indicates that there was only a 5% chance of a false positive and a 5% chance of a false positive.')
        st.image('Screenshot 2022-05-04 123412.png')


