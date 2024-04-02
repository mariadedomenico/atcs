import main
import pandas as pd
import random

def predictionTest(users, movies, method):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    n_users = df['userId'].nunique()

    accuracy = 0
    for u in users:
        df1 = df[df['userId'] == u]

        topUsers = main.getTopKUsers(u, n_users, df, method)

        neighbors = main.makeMovieDict(df, topUsers, movies[u], 40)

        preds = 0
        for m in movies[u]:
            prediction = round(main.prediction(df, df1, m, neighbors[m], 'test'), 1)
            actual = df1.loc[df1['movieId'] == m, 'rating'].values[0]
            # print(prediction, actual)
            if abs(prediction-actual) <= 1:
                preds+=1

        accuracy += preds/len(movies[u])

    print(round(accuracy/len(users), 1))


def makeData():

    df = pd.read_csv('ml-latest-small/ratings.csv')

    random_users = random.sample(df['userId'].unique().tolist(), 5)

    users_movies = {}

    for u in random_users:
        movies = df[df['userId'] == u]['movieId'].unique().tolist()
        random_movies = random.sample(movies, min(10, len(movies)))
        users_movies[u] = random_movies

    return (random_users, users_movies)

(users, movies) = makeData()
print(users)
print(movies)
predictionTest(users, movies, main.similarityPearson)
predictionTest(users, movies, main.cosineSimilarity)
predictionTest(users, movies, main.jaccardReal)
predictionTest(users, movies, main.jaccardCustom)
predictionTest(users, movies, main.improvedTraingleSimilarity)
