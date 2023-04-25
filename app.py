# flask app

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from model import recommend_article, PopularityModel, ContentBaseModel, NMF_recommendation, get_sparse_matrix, NMFModel
from blob import blobConn
from constant import nmf_params

app = Flask(__name__)

# load previous calculated data and model from Azure
# latest_clicks file are the click activities in the past day
latest_clicks = blobConn().download('rec-model-v1','latest_clicks.csv','csv')
# eb file is the pretrained embeddings used for content based model
eb = blobConn().download('rec-model-v1','pca_article_embeddings.pickle','pickle')
# nmf_train is user, article and click information used as input for NMF model traininig. 
# It has user in [0, 10000] and article in [0, 10000]
nmf_train = blobConn().download('rec-model-v1','nmf_train2.csv','csv')
# all_user_article_interaction has all the clicks data. 
# Used for retrieving user history if the user doesn't in nmf training data.
all_user_article_interaction = blobConn().download('rec-model-v1','all_user_article_interaction.csv','csv')



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommendations',methods=['POST'])
def recommendations():
    # Get the user input from the form
    user_id = request.form['user_id']
    country_id = request.form['country_id']
    region_id = request.form['region_id']
    # Get article recommendations from content-based model
    content_based_recs = ContentBaseModel(latest_clicks, 
                                          eb, 
                                          user_id, 
                                          country_id, 
                                          region_id)

    # Get article recommendations from collaborative filtering model

    R_train = get_sparse_matrix(nmf_train, shape = [10000,10000])
    R_pred, H, estimator = NMFModel(R_train, **nmf_params)
    collab_filtering_recs = NMF_recommendation(all_user_article_interaction, 
                                               latest_clicks,
                                               estimator, 
                                               H, 
                                               R_pred, 
                                               user_id, 
                                               country_id, region_id)
    #if collab_filtering_recs == '404':
    #    collab_filtering_recs = 'completely new user without any history, use popularity model'

    # Render the recommendations template with the recommendation results
    return render_template('recommendations.html',
                           content_based_recs=content_based_recs,
                           collab_filtering_recs=collab_filtering_recs)




if __name__ == '__main__':
    app.run(debug=True)





