

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
#from sklearn.metrics import mean_squared_error

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


def PopularityModel(df,country,region):
    '''read most recent 24 hour data'''
    country = int(country)
    region = int(region)
    df2 = df[(df['click_country']==country) & (df['click_region']==region)]
    if len(df2) == 0:
        df2 = df[df['click_country']==country]
        output = df2.groupby(['click_article_id'],as_index = False).size().nlargest(5,'size').reset_index(drop=True)
    else:
        output = df2.groupby(['click_article_id'],as_index = False).size().nlargest(5,'size').reset_index(drop=True)
    return output.click_article_id.values

def ContentBaseModel(data, embedding, user_id, country, region):
    # find most recent articles and recommend 5 use pre trained embeddings
    user_id = int(user_id)
    data_sub = data[data['user_id']==user_id]
    if len(data_sub) == 0: # new user
        rec = PopularityModel(data,country, region)
        if len(rec) == 0:
            return '404'
    else:
        most_recent_article = data_sub.loc[data_sub['click_timestamp']==np.max(data_sub['click_timestamp']),'click_article_id']
        rec = recommend_article(embedding, most_recent_article,5)
    return rec


def get_sparse_matrix(data, shape=None):
    """data is the user article click dataframe"""
    row = data.iloc[:,0].values
    col = data.iloc[:,1].values
    val = data.iloc[:,2].values
    n_user = len(data.user_id.unique())
    n_article = len(data.click_article_id.unique())
    max_user_idx = max(row)
    max_article_idx = max(col)
    if not shape:
        sparse_mat = sparse.csc_matrix((val, (row, col)), shape=(max_user_idx+1, max_article_idx+1))

    else:
        sparse_mat = sparse.csc_matrix((val, (row, col)), shape=(shape[0], shape[1]))

    return sparse_mat


def NMFModel(R, **params_NMF):
    """R is sparse matrix"""
    model = NMF(**params_NMF)
    W = model.fit_transform(R)
    #H = model.components_
    R_hat = model.inverse_transform(model.transform(R))
    return R_hat, model


def NMF_recommendation(data, estimator, R_train, R_pred, user_id):
    """R_pred is array
        data is interaction data generated already
    """
    user_id = int(user_id)
    if user_id < R_pred.shape[0]:
        # user in prediction matrix
        sort_article = np.argsort(R_pred[user_id])[::-1][:5]
    else:
        # user not in prediction matrix (0-10000)
        # add it to train matrix to retrain the model
        history = np.full((1,R_pred.shape[1]),0)
        user_df = data[(data['user_id'] == user_id) & (data['click_article_id']<=R_pred.shape[1])]
        if len(user_df) == 0:
            #completely new with no click history
            return '404'
        indexs = user_df.click_article_id.values
        print(history.shape)
        history[0][indexs] = 1
        new_R = np.concatenate((R_train.toarray(),history),axis = 0)
        new_R_hat = estimator.inverse_transform(estimator.transform(new_R))
        sort_article = np.argsort(new_R_hat[-1])[::-1][:5]

    return sort_article



"""
if __name__=="__main__":
    eb = pd.read_pickle("pre_trained/articles_embeddings.pickle")
    id=12303
    result = recommend_article(eb,id,5)
    print(result)
"""
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

