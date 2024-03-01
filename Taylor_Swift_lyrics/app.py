import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pysankey2 import Sankey
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
from wordcloud import STOPWORDS
from plotly.subplots import make_subplots
import string 
import re
import plotly.figure_factory as ff
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set the title of the web app
st.set_page_config(
    page_title='Taylor Swift in Data',
    page_icon='🦋',
    layout='centered',
)
#dark background
@st.cache_data
def load_data():
    data = pd.read_csv('taylorswiftspotify.csv',encoding='latin1')
    return data

data = load_data()

# Set the title of the web app\
st.title('Taylor Swift: Through the Lens of Data')
st.write('The aim of this analysis is to have a more in dept look at her discography and songwriting, with a special attention to the results achieved in terms of Spotify streams and pure sales.')

header = '''
## General Overview
'''
st.markdown(header)
header_txt = '''### The musical genres of Taylor Swift'''
st.markdown(header_txt)
st.write('Let us explore the main musical genres that she has explored in her career')
fig = go.Figure(data=[go.Pie(labels=data['genre1'].value_counts().index, values=data['genre1'].value_counts().values,hole=.5)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=['#FFD700', '#C0C0C0', '#CD7F32', '#FF6347', '#00FFFF', '#FF00FF', '#00FF00'], line=dict(color='#000000', width=2)))
st.plotly_chart(fig, use_container_width=True)

st.write(f'From this donut chart, we can see that 45.51% of Swift’s discography is made of pop songs, 32.69% is made of country songs and the remaining 21.79% is made of alternative songs.')

album_header = '''### Albums and type of writing'''
st.markdown(album_header)
st.write(f'Let’s see more in particular which are the albums that are respectively alternative, country or pop and in which way these genres are associated to the types of writing.')

df_sankey = data[['genre1','album_name','rating','writing']]

nodes = pd.unique(df_sankey[['genre1', 'album_name', 'rating', 'writing']].values.ravel('K'))
nodes_dict = {node: idx for idx, node in enumerate(nodes)}

source_indices_genre = [nodes_dict[genre] for genre in df_sankey['genre1']]
target_indices_album = [nodes_dict[album] for album in df_sankey['album_name']]
source_indices_album = [nodes_dict[album] for album in df_sankey['album_name']]
target_indices_rating = [nodes_dict[rating] for rating in df_sankey['rating']]
source_indices_rating = [nodes_dict[rating] for rating in df_sankey['rating']]
target_indices_writing = [nodes_dict[writing] for writing in df_sankey['writing']]

value = [1] * len(source_indices_genre + source_indices_album + source_indices_rating)
source_indices = source_indices_genre + source_indices_album + source_indices_rating
target_indices = target_indices_album + target_indices_rating + target_indices_writing

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=nodes
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=value
    )
)])

# Add labels below each column
fig.add_annotation(
    x=0,
    y=-0.1,
    xref='paper',
    yref='paper',
    showarrow=False,
    text='Genre',
    font=dict(size=12)
)

fig.add_annotation(
    x=0.35,
    y=-0.1,
    xref='paper',
    yref='paper',
    showarrow=False,
    text='Album',
    font=dict(size=12)
)

fig.add_annotation(
    x=0.70,
    y=-0.1,
    xref='paper',
    yref='paper',
    showarrow=False,
    text='Rating',
    font=dict(size=12)
)

fig.add_annotation(
    x=1.0,
    y=-0.1,
    xref='paper',
    yref='paper',
    showarrow=False,
    text='Writing',
    font=dict(size=12)
)

# Update layout for better visibility
fig.update_layout(title_text="The albums by genre and writing",
                  font_size=10,
                  autosize=True)

# Show the plot
st.plotly_chart(fig, use_container_width=True)

sankey_caption = '''
We can see that the country albums are Taylor Swift (2006), Fearless (2008) and Speak Now (2010) while, the pop ones are Red (2012), 1989 (2014), Reputation (2017) and Lover (2019) while the alternative ones are Folklore (2020) and Evermore (2020). A first interesting aspect to point out is that the singer has explored each genre with different consecutive albums before switching to the next one, to consolidate her ability in that genre. Moreover, only with her most recent alternative albums (at the age of 31), she has made “abuse” of explicit lyrics while the country albums are completely clean and only a very small number of pop songs are explicit. However, all the explicit songs are co-written. This probably means that she doesn’t write explicit lyrics herself but her co-writers do and she only accepts to put them in her songs.'''
st.markdown(sankey_caption)

spotify_header = '''## Spotify features'''
st.markdown(spotify_header)
st.write('There are a lot of numerical variables that describe Taylor’s songs and her way of making country, pop and alternative music. It may be dispersive to look at all of them. Let’s start by looking at the most important ones, the most correlated ones.')

num_features = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','tempo','valence']
corr = data[num_features].corr()

fig = px.imshow(corr, text_auto=".2f",color_continuous_scale='RdBu', title='Correlation matrix of the numerical features',aspect='auto')
st.plotly_chart(fig, use_container_width=True)

spotify_caption = '''From the correlogram, the positive correlations that stand out the most are those between:
- energy and loudness
- danceability and valence'''
st.markdown(spotify_caption)

feature_header = '''### Danceability and Valence'''
st.markdown(feature_header)

fig = px.density_heatmap(data, x="danceability", y="valence", marginal_x="histogram", marginal_y="histogram", title='Danceability and Valence')
st.plotly_chart(fig, use_container_width=True)
density_caption = '''This plot better displays the positive relationship between the two variables that was observed in the correlation matrix: as danceability increases, valence increases too (and viceversa). Moreover, it is clear from the plot that most of Taylor’s songs have high values for these variables, both between 0.5 and 0.7.'''
st.markdown(density_caption)

feature_header2 = '''### Energy and Loudness'''
st.markdown(feature_header2)
fig = px.density_heatmap(data, x="energy", y="loudness", marginal_x="histogram", marginal_y="histogram", title='Energy and Loudness')
st.plotly_chart(fig, use_container_width=True)
density_caption2 = '''This plot  displays the positive relationship between the two variables that was observed in the correlation matrix: as energy increases, loudness increases too (and viceversa). Moreover, it is clear from the plot that most of Taylor’s songs have high values for these variables, both between 0.6 and 0.7.'''
st.markdown(density_caption2)

header3 = ''' ## Text Analysis'''
st.markdown(header3)
st.write('Now let us explore the lyrics of her songs')

lyrics_df = data[['track_name','album_name','lyrics']]
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]','',text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub(r'\w*\d\w*','',text)
    return text
lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(lambda x: clean_text(x))
lyrics_df['tokens'] = lyrics_df['lyrics'].apply(lambda x: len(nltk.word_tokenize(x)))
data['album_year'] = pd.DatetimeIndex(data['album_release']).year
lyrics_df['album_year'] = data['album_year']
def lexical_diversity(text):
    return len(set(text)) / len(text)

lyrics_df['lexical_diversity'] = lyrics_df['lyrics'].apply(lambda x: lexical_diversity(nltk.word_tokenize(x)))
st.markdown('''### Lyrics diversity in Taylor's songs''')
st.write('Taylor Swift is known not for having an impressive voice, but for her brilliant songwriting and storytelling to which people can relate. In 2010 she was named country songwriter of the year at the BMI Awards, becoming the youngest person to win the award at age 20. In the same year she received the prestigious Hal David Starlight Award at the Songwriters Hall of Fame Awards, joining legends like Alicia Keys, John Mayer and John Legend. Most recently, she was awarded with the 2021 National Music Publishers’ Association’s Songwriter Icon Award.')
st.write(lyrics_df.head())

fig = ff.create_distplot([lyrics_df['tokens']], group_labels=['lyrics'],bin_size=10, show_rug=True)# green - #00FF00
#add x and y axis labels
fig.update_layout(xaxis_title='Tokens', yaxis_title='Density')
fig.update_layout(title_text='Distribution of tokens in Taylor Swift songs')
#draw a vertical line at the mean
#fig.add_vline(x=lyrics_df['tokens'].mean(), line_dash="dash", line_color="black", annotation_text="Mean", annotation_position="top right")
st.plotly_chart(fig, use_container_width=True)
token_caption = '''The table and the plot show that, indeed, her songs contain a lot of lyrics. All of her songs present more than 150 words and, actually, some of them exceed 500 words, even though the greatest part of her songs contain between 300 and 450 words. As a result, the average number of words per song is around 368, which is not low at all.'''
st.markdown(token_caption)

fig = px.bar(lyrics_df, x='album_year', y='tokens',hover_data=['track_name'], labels={'tokens':'Number of words'}, title='Number of words in Taylor Swift songs over the years')
#adjust the x axis to show all the years
fig.update_xaxes(tickmode='linear')
st.plotly_chart(fig, use_container_width=True)
st.write('The plot shows the average number of words per song that she has written in a given year. When Swift started her career way back in 2006 with the self-titled album, she wrote, on average, 267 words per song. This average increased until 2017 (with a small decrease in 2012 for the album Red) but with the last three albums, Lover (2019), Folklore (2020) and Evermore (2020) it has started to decrease more evidently, probably due to the fact that she has shifted to the alternative genre, which is more focused on the melody rather than on using a lot of words.')

fig = px.bar(lyrics_df, x='album_year', y='lexical_diversity',hover_data=['track_name'], title='Lexical diversity in Taylor Swift songs over the years')
#adjust the x axis to show all the years
fig.update_xaxes(tickmode='linear')
st.plotly_chart(fig, use_container_width=True)
st.write('The plot shows the lexical diversity of her songs over the years. The lexical diversity is the ratio of the number of unique words to the total number of words in a text. The higher the ratio, the more diverse the vocabulary. The highest value of the type-token ratio is reached in 2006, when actually Miss. Swift started her career. This means that her first album has the lowest average number of words pr song, but at least they are not always the same words. Then, there was a period (2012-2017) in which it was very low and this corresponds also to the period of pop songs, which have catchy but repeated chorus. In the last years we can see that this ratio is going back to its origin and could increase in the future.')

word_header = '''### Wordcloud of Taylor Swift songs'''
st.markdown(word_header)
st.write('Let us see the most frequent words in her songs')
text = " ".join(lyrics_df['lyrics'])

wordcloud = WordCloud(width=800,height=400,background_color='black',stopwords=STOPWORDS).generate(text)

fig = plt.figure(figsize=(10,5))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)
st.write('As expected from a songwriter who writes about her personal life and experiences, the most common theme is that of love.')

header4  = '''## Album Sales'''
st.markdown(header4)
st.write('Let us see the pure sales of her albums')
st.write(f'Swift is known for being one of the best-selling musicians of all time. When she releases new music, she always tops the chart.')

def sales_data():
    sales = pd.read_csv('albumsales.csv')
    return sales

sales = sales_data()
puresales = sales.groupby('Album')['Sales'].sum().reset_index()
Year = ["2014", "2020", "2008", "2020", "2019", "2012", "2017", "2010", "2006"]
puresales['Year'] = Year
puresales = puresales[['Album', 'Year', 'Sales']]
puresales = pd.DataFrame(puresales)
puresales = puresales.sort_values(by=['Year', 'Album'], ascending=[True, False])
puresales["Type"] = "Pure Sales"
puresales["year_parenthesis"] = "(" + puresales["Year"] + ")"
puresales["album_year"] = puresales["Album"] + " " + puresales["year_parenthesis"]

fig = px.treemap(puresales, path=['Type', 'album_year'], values='Sales', title='Pure sales of Taylor Swift albums')
fig.data[0].textinfo = 'label+text+value'
#increase the size of the plot
#fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)
st.write('1989 is Swift’s most sold album. Followed by Fearless and Speak Now. The last three albums, Lover, Folklore and Evermore, have sold less than the previous ones, but they are still in the top 10 of her most sold albums.')

header5 = '''## 1989 Sales'''
def sales_data2():
    sales2 = pd.read_csv('album1989dailystreams.csv')
    return sales2

sales_1989 = sales_data2()
st.markdown(header5)
st.write(f'As 1989 is Swift’s most sold album, let’s see if listening patterns')
sales_1989['Week'] = sales_1989['Week'].astype('str')
sales_1989['Date'] = pd.to_datetime(sales_1989['Date'],format="%Y-%m-%d")
day_mapping = {"Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu", "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun"}
sales_1989["Day_week"] = sales_1989["Day_week"].map(day_mapping).astype("category")
sales_1989["Daily_streams"] = pd.to_numeric(sales_1989["Daily_streams"])
filtered_album1989dailystreams = sales_1989[sales_1989["Week"].isin(["11", "12", "13", "14", "15", "16"])]
filtered_album1989dailystreams["Daily_streams"] = filtered_album1989dailystreams["Daily_streams"] - 3000000

new_album1989dailystreams = filtered_album1989dailystreams.pivot_table(values="Daily_streams", columns="Day_week", index="Week").reset_index()

categories = new_album1989dailystreams.columns[1:]
num_categories = len(categories)

angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

for i, (label, row) in enumerate(new_album1989dailystreams.iterrows()):
    values = row.values[1:].tolist()
    values += values[:1]
    ax.plot(angles, values, label=f"Week {label}", linewidth=2, linestyle='solid', alpha=0.7)
    #ax.fill(angles, values, alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_rlabel_position(0)
plt.yticks([500000, 1000000], ["500k", "1M"], color="black", size=8)
plt.ylim(0, 1000000)

plt.title("1989's daily streams", size=13, y=1.1)
plt.suptitle("from 14th march to 24th april 2022", size=8, y=0.94)
#make the legend smaller
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=8)
st.pyplot(fig)
st.write('the streams tend to be lower on Saturdays and especially on Sundays, probably because people go out and do not have a lot of time to listen to music. Even if they stay at home, they may decide to use physical supports because they have more time. Instead, the streams are higher in the workdays and this could be due to the fact that when people go to work or to school they listen to music')
