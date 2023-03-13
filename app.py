# flask app

from flask import Flask, request, jsonify
import pandas as pd
from model import find_top_n, recommend_article

app = Flask(__name__)

def get_pickle():
    df = pd.read_pickle("pre_trained/articles_embeddings.pickle")
    return df

def generate_recommendations(article_id):
    # Your code here to generate recommendations based on the book ID
    df = get_pickle()
    recommendations = recommend_article(df,article_id,5)
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
    recommendations = generate_recommendations(article_id)
    # Convert the recommendations to a JSON response
    response = jsonify(recommendations)
    return response



if __name__ == '__main__':
    app.run(debug=True)





