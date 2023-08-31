import os

from flask import Flask

import json
from sqlalchemy import create_engine

import plotly
import pandas as pd
import re
from wordcloud import WordCloud
import plotly.graph_objects as go

import base64
import io
import tempfile

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import sklearn.externals
import joblib
import plotly.express as px




app = Flask(__name__)

# load data 
engine = create_engine('sqlite:///app/data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) #removing all punctuations
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




# load model
model = joblib.load("classifier.pkl")

color_palette = {
    'direct': '#999999',
    'news': '#e41a1c',
    'social': '#dede00'
}


@app.route("/")
@app.route('/index')
def index():
  
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # barplot 1: number of messages by genre
    #x: genre names y: genre counts

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #heatmap by category

    categories = df.columns[4:] 
    messages = df['id']

    #making the heatmap
    heatmap_fig = px.imshow(df[categories], x=categories, y=messages, color_continuous_scale=[[0, 'black'], [1, 'pink']])

    #heatmap layout
    heatmap_fig.update_layout(
    xaxis_title="Categories",
    yaxis_title="Messages",
    title="Distribution of messages by category"
)



    #subsetting data for each genre
    subset_direct = df[df['genre'] == 'direct']
    subset_news = df[df['genre'] == 'news']
    subset_social = df[df['genre'] == 'social']

    # create wordclouds
    wc_direct = WordCloud(background_color='#999999').generate(str(subset_direct['message'].values))
    wc_news = WordCloud(background_color='#e41a1c').generate(str(subset_news['message'].values))
    wc_social = WordCloud(background_color='#dede00').generate(str(subset_social['message'].values))

    with tempfile.NamedTemporaryFile(suffix='.png') as direct_file, \
            tempfile.NamedTemporaryFile(suffix='.png') as news_file, \
            tempfile.NamedTemporaryFile(suffix='.png') as social_file:
        wc_direct.to_file(direct_file.name)
        wc_news.to_file(news_file.name)
        wc_social.to_file(social_file.name)

        direct_img_base64 = base64.b64encode(direct_file.read()).decode('utf-8')
        news_img_base64 = base64.b64encode(news_file.read()).decode('utf-8')
        social_img_base64 = base64.b64encode(social_file.read()).decode('utf-8')



    # TODO: Below is an example - modify to create your own visuals
    graphs = [
    heatmap_fig,
    {
        'data': [
            {
                'x': genre_names,
                'y': genre_counts,
                'type': 'bar',
                'marker': {'color': [color_palette.get(genre, 'viridis') for genre in genre_names]}
            }
        ],

        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }

    ]

    

  
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, direct_img=direct_img_base64, news_img=news_img_base64, social_img=social_img_base64)



 #web page that handles user query and displays model results
@app.route('/go')
def go():
   # save user input in query
   query = request.args.get('input', '') 

   # use model to predict classification for query
   classification_labels = model.predict([query])[0]
   classification_results = dict(zip(df.columns[4:], classification_labels))

   # This will render the go.html Please see that file. 
   return render_template(
       'go.html',
       query=query,
       classification_result=classification_results
   )

if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


