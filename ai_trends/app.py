from turtle import title
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

st.set_page_config(
    page_title='AI Trends',
    page_icon='📊',
    layout='wide',
)
@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

data = load_data('ai_trends/ai-performance-knowledge-tests-vs-training-computation.csv')
st.title('Key insights on Artificial Intelligence in recent years')
st.markdown('This is a dashboard that provides insights on AI-related metrics that let you monitor what is happening and where we might be heading')
header = '''
## Performance on knowledge tests vs. training computation
'''
st.markdown(header)
#st.write("Training computation is measured in total floating point operations, or “FLOP” for short. One FLOP is equivalent to one addition, subtraction, multiplication, or division of two decimal numbers.")
#st.write("The chart below shows The chart shows that over the last decade, the amount of computation used to train the largest AI systems has increased exponentially.")
fig = px.scatter(data,x='Training computation (petaFLOP)',y='MMLU avg',color='Organization',hover_data=['Entity'],log_x=True,title='AI Performance in Knowledge Tests vs Training Computation')
fig.update_traces(textposition='top center')
#add vertical gridlines
fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
st.plotly_chart(fig, use_container_width=False)
st.write("Each dot on this chart represents a distinct language model. The horizontal axis shows the training computation used (on a logarithmic scale), measured in total floating point operations (“FLOP”). The vertical axis indicates the model's performance on the Massive Multitask Language Understanding (MMLU) benchmark, an extensive knowledge test composed of thousands of multiple-choice questions across 57 diverse subjects, from science to history. As training computation has risen, so has performance on these knowledge tests.")

header = '''
## Exponential increase in the computation used to train AI
'''
st.markdown(header)
train_compute = load_data('ai_trends/artificial-intelligence-training-computation.csv')
train_compute = train_compute.drop(['Code'],axis=1)
train_compute['year'] = pd.DatetimeIndex(train_compute['Day']).year
fig = px.scatter(train_compute,x='year',y='Training computation (petaFLOP)',color='Domain',log_y=True,hover_data=['Entity'],title='Exponential increase in the computation used to train AI')
fig.update_traces(textposition='top center')
fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
st.plotly_chart(fig, use_container_width=False)
st.write("Each small circle on this chart represents one AI system. The circle’s position on the horizontal axis indicates when the AI system was made public, and its position on the vertical axis shows the amount of computation used to train it. It’s shown on a logarithmic scale. The chart shows that over the last decade, the amount of computation used to train the largest AI systems has increased exponentially. ")

header='''
## Corporate investment in AI
'''
st.markdown(header)
corpo = load_data('ai_trends/corporate-investment-in-artificial-intelligence-by-type.csv')
fig = px.bar(corpo,x='Year',y='Total corporate investment - inflation adjusted',color='Entity')
fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
st.plotly_chart(fig, use_container_width=False)
st.write("Investments in 2021 were about 30 times larger than a decade earlier. Given how rapidly AI developed in the past – despite its limited resources – we might expect AI technology to become much more powerful in the coming decades, now that the resources dedicated to its development have increased so substantially.")

header='''
## AI hardware production
'''
st.markdown(header)
header_txt = '''### Design'''
st.markdown(header_txt)
#make three columns
share = load_data('ai_trends/market-share-logic-chip-production-manufacturing-stage.csv')
share = share.drop(['Code'],axis=1)
share = share.dropna()
st.write("More than 90% of these chips are designed and assembled in only a handful of countries: the United States, Taiwan, China, South Korea, and Japan.")
fig = px.bar(share,y='Entity',x='Design',text_auto='.2s',title='Design',color='Entity')
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=False)

header_txt = '''### Fabrication'''
st.markdown(header_txt)
st.write("The fabrication stage is where the silicon wafer is manufactured. The fabrication stage is the most expensive part of the chip-making process. Taiwan's TSMC and South Korea's Samsung are the two largest companies in this stage.")
fig = px.bar(share,y='Entity',x='Fabrication',text_auto='.2s',title='Fabrication',color='Entity')
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=True)

header_txt = '''### Assembly, testing, and packaging'''
st.markdown(header_txt)
st.write("The assembly, testing, and packaging stage is where the chip is put into a package and tested. This stage is less expensive than the fabrication stage. The largest companies in this stage are in Taiwan, China, and the United States.")
fig = px.bar(share,y='Entity',x='Assembly, testing and packaging',text_auto='.2s',title='Assembly, testing and packaging',color='Entity')
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=False)

st.write("AI tends to focus on software and algorithmic improvements, a few countries could, therefore, dictate the direction and evolution of AI technologies through their influence on hardware.")
