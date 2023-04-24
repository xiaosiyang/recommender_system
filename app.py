# flask app

from flask import Flask, request
import pandas as pd
import numpy as np
from model import recommend_article, PopularityModel, ContentBaseModel
from blob import blobConn

app = Flask(__name__)


latest_clicks = blobConn().download('rec-model-v1','latest_clicks.csv','csv')
eb = blobConn().download('rec-model-v1','pca_article_embeddings.pickle','pickle')

# previous version
def generate_recommendations(embedding,article_id):
    # Your code here to generate recommendations based on the book ID
    recommendations = recommend_article(embedding,article_id,5)
    return recommendations

'''
@app.route('/')
def home():
    return 'hello world'

'''

@app.route('/')
def home():
    return '''
         <form method="POST" action="/recommendations">
            <label for="user-id">Enter User ID:</label>
            <input type="text" id="user-id" name="user_id">
            <label for="country-id">Enter Country ID:</label>
            <input type="text" id="country-id" name="country_id">
            <label for="region-id">Enter Region ID:</label>
            <input type="text" id="region-id" name="region_id">            
            <button type="submit">Get recommendations</button>
        </form>   
    '''


'''
@app.route('/recommendations', methods=['POST'])
def recommendations():
    if user_id in train:

        article_id = request.form['article_id']
        # Generate recommendations based on the book ID
        recommendations = generate_recommendations(eb,article_id)
        # Convert the recommendations to a JSON response
        #response = jsonify(recommendations)
    elif user_id not in and user_id has history:
        recent_article = get_most_recent_article()
        if recent_article in eb:
            generate_recommendations(eb, article_id)
        else:
            content_based_article_meta_data()
    else: # complete new user
        popularity_model()
'''

@app.route('/recommendations', methods=['POST'])
def recommendations():       
    user_id = request.form['user_id']
    country_id = request.form['country_id']
    region_id = request.form['region_id']

    recommendations = ContentBaseModel(latest_clicks, eb, user_id, country_id, region_id)
    if recommendations == '404':
        html = '<label>No data available</label>'
    else:
        html = '<table><tr><th>Rank</th><th>Recommended Article ID</th></tr>'
        for id, article in enumerate(recommendations):
            nid = id+1
            html += f'<tr><td>{nid}</td><td>{article}</td></tr>'
        html += '</table>'
        
        # Add a "go back" button to return to the home page
    html += '<br><button onclick="location.href=\'/\'">Go Back</button>'
    
    # Return the HTML page with the recommendations and "go back" button
    return html



if __name__ == '__main__':
    app.run(debug=True)





