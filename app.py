# flask app

from flask import Flask, request, jsonify
import pandas as pd
from model import find_top_n, recommend_article
from blob import blobConn

app = Flask(__name__)

def get_pickle_local():
    df = pd.read_pickle("pre_trained/articles_embeddings.pickle")
    return df

def get_pickle_az():
    array = blobConn().download('rec-model-v1','pre-trained-embeddings')
    return array

eb = get_pickle_az()

def generate_recommendations(embedding,article_id):
    # Your code here to generate recommendations based on the book ID
    recommendations = recommend_article(embedding,article_id,5)
    return recommendations

@app.route('/')
def home():
    return '''
         <form method="POST" action="/recommendations">
            <label for="article-id">Enter Ariticle ID:</label>
            <input type="text" id="article-id" name="article_id">
            <button type="submit">Get recommendations</button>
        </form>   
    '''


@app.route('/recommendations', methods=['POST'])
def recommendations():
    article_id = request.form['article_id']
    # Generate recommendations based on the book ID
    recommendations = generate_recommendations(eb,article_id)
    # Convert the recommendations to a JSON response
    #response = jsonify(recommendations)

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





