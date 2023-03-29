

import numpy as np
import pandas as pd
import os
import glob


def find_top_n(data, n):
    """Return top n articles from a list of similarity values
    params
    data: list
    id: int
    """
    idx_data = enumerate(data)
    sorted_data = sorted(idx_data, 
                         key=lambda x: x[1], 
                         reverse=True)
    return [id[0] for id in sorted_data[:n]]

def recommend_article(embedding, article_id, n):
    """calculate similarities and give recommendation
    Params
    embedding: embedding list
    article_id: int
    n:int
    """
    score = []
    for idx in range(len(embedding)):
        #print(idx,article_id)
        a_id = int(article_id)
        if idx != a_id:
            simScore = np.dot(embedding[idx],embedding[a_id])/(np.linalg.norm(embedding[idx])*np.linalg.norm(embedding[a_id]))
            score.append(simScore)
    rec = find_top_n(score,n)
    return rec


def PopularityModel(data):
    '''raw_data is the click data
    read most recent 24 hour data'''
    lc = data.groupby(['click_article_id'],as_index = False).size().nlargest(5,'size').reset_index(drop=True)
    return lc.click_article_id.values

def ContentBaseModel(data, embedding, user_id):
    # find most recent articles and recommend 5 use pre trained embeddings
    data_sub = data[data['user_id']==user_id]
    if len(data_sub) == 0: # new user
        rec = PopularityModel(data)
    else:
        most_recent_article = data_sub.loc[data_sub['click_timestamp']==np.max(data_sub['click_timestamp']),['click_article_id']]
        rec = recommend_article(embedding, most_recent_article,5)
    return rec



if __name__=="__main__":
    eb = pd.read_pickle("pre_trained/articles_embeddings.pickle")
    id=12303
    result = recommend_article(eb,id,5)
    print(result)

"""
# for a user not previously incorporated into the model, 
# run the new user's interaction through the matrix factorization model, 
# and estimate the latent factors
"""

# input user id (log in) 
# list user id range, env, device_os, country

# New user that has no article reading history
## recommend the most popular articles trending
### once he clicked some article, then there's data, 
### we can use content based model to recommend to the user 
# with articles that are similar with the one he clicked
## the matrix factorizaton model could also be retrained with new user added in


# New articles added in

