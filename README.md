## About the project
This is a project using Kaggle data. Details of the data can be find [here](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom).
The goal is to create a article recommender system and show 5 articles for each user.

## Method
The recommender system includes three models
- Content based model
- Non negative matrix factorization
- Popularity model

Content based model use pre trained embeddings provided in the project. PCA is used to reduce the dimension of embeddings to 30. The algorithm recommends articles based on the user's **most recent article click in past day** and recommend the 5 most similar articles to the user's last clicked articles. If the user doesn't have click history in the last day, then popularity model will be used.

NMF is an alternative algorithm to content based model. 
Data transformation: user, article, clicks, clip clicks > 1 to be 1 since we only care about if the user clicks the article, not how many times.

Due to the CPU/memory limitation on my own laptop, I was only able to train on a small set of parameters, so the NMF model performance is not good.

Popularity model is used when the previous two models don't work. The system will recommend the top 5 articles in the country/region the user is from. If the region doesn't have this service yet, the algorithm just recommend top 5 most popular articles in the country.

## Performance
The model is trained on 10K users and 10K articles.
Best model is selected by
- 4-fold cross validation
- Evaluation metrics: RMSE


## Architecture
![alt text](https://github.com/xiaosiyang/recommender_system/blob/main/resource/arch_v3.svg)


## Test Example

Existing user
- content based model covers user_id in [0, 322896], article_id in [0,364046]
    - example: user 3066
- NMF model covers user_id in [0, 10000], article_id in [0,10000]
    - example: user 322842, NMF model will compute user's preference and apply trained model. The NMF recommendations will be article in [0,10000], content based model will give different recommendations.
    - example: user 23, who is already in the pretrained matrix, quick result.
New user without any history
- example: user 999999, popularity based model will generate the same result