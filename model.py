

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

if __name__=="__main__":
    eb = pd.read_pickle("pre_trained/articles_embeddings.pickle")
    id=12303
    result = recommend_article(eb,id,5)
    print(result)

