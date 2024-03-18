import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce

### TO RUN TOP 10 MOVIES, UNCOMMENT IF NAME == MAIN AND COMMENT RUNMETHODS. CHANGE GROUP IN MAIN###
### TO RUN DIFFERENT METHODS, COMMENT IF NAME == MAIN AND UNCOMMENT RUNMETHODS. CHANGE GROUP BY PARAM ###

def similarityPearson(df1, df2) :

    movie_set1 = set(df1['movieId'].tolist())
    movie_set2 = set(df2['movieId'].tolist())
    movie_intersection = movie_set1.intersection(movie_set2)
    media_ratings1 = df1['rating'].mean()
    media_ratings2 = df2['rating'].mean()
    numerator = 0
    denominator_sum1 = 0
    denominator_sum2 = 0

    for movie in movie_intersection :
        rating1 = df1.loc[df1['movieId'] == movie, 'rating'].values[0]
        rating2 = df2.loc[df2['movieId'] == movie, 'rating'].values[0]
        firstElem = rating1 - media_ratings1
        secondElem = rating2 - media_ratings2
        numerator = numerator + (firstElem * secondElem)
        denominator_sum1 = denominator_sum1 + (firstElem ** 2)
        denominator_sum2 = denominator_sum2 + (secondElem ** 2)

    denominator = np.sqrt(denominator_sum1) * np.sqrt(denominator_sum2)
    if denominator == 0 :
        return 0
    else:
        similarity = numerator/denominator
        return similarity


def prediction(df, df1, movieId) :

    df_elem = df[df['movieId'] == movieId]
    if movieId in df1['movieId'].tolist():
        return df.loc[df['movieId'] == movieId, 'rating'].values[0]
    
    avg_ratings1 = df1['rating'].mean()
    num = 0
    similaritySum = 0
    i = 0

    for _, row in df_elem.iterrows(): 
        df2 = df[df['userId'] == row['userId']]
        sim = similarityPearson(df1, df2)
        if sim>0 and i<40:
            avg_ratings2 = df2['rating'].mean()
            num = num + (sim * (row['rating'] - avg_ratings2))
            similaritySum = similaritySum + sim
        elif i >= 40:
            break

    if(similaritySum != 0):
        prediction = avg_ratings1 + (num/similaritySum)
    else:
        prediction = avg_ratings1
    return prediction


def weightedAverageMethod(group, movie):
    
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = set()
    for user in group:
        user = df[df['userId'] == user]
        rating = prediction(df, user, movie)
        print(rating)
        ratings.add(round(rating,2))
    ratings_sum = sum(ratings)
    pred = ratings_sum / len(ratings)
    return round(pred,2)

def averageMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    for user in group:
        user = df[df['userId'] == user]
        rating = prediction(df, user, movie)
        print(rating)
        ratings.append(round(rating,2))
    ratings_sum = sum(ratings)
    pred = ratings_sum / len(ratings)
    return round(pred,2)

def leastMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    for user in group:
        user = df[df['userId'] == user]
        rating = prediction(df, user, movie)
        ratings.append(round(rating,2))
    return round(min(ratings), 2)

def avgAndLeastMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    for member in group:
        user = df[df['userId'] == member]
        rating = prediction(df, user, movie)
        ratings.append(round(rating,2))
    ratings_sum = sum(ratings)
    least = min(ratings)
    avg = ratings_sum / len(ratings)
    return (round(avg,2), round(least, 2))


def calculate_prediction(args):
    group, movies = args
    predictionsAvg = []
    predictionsLeast = []
    for movie in movies:
        print(movie)
        pred = avgAndLeastMethod(group, movie)
        predictionsLeast.append((movie, pred[1]))
        predictionsAvg.append((movie, pred[0]))
    return [predictionsLeast, predictionsAvg]

def checkPredictionPar(group):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    setTot = set(df['movieId'].tolist())
    member_list = []
    for member in group:
        user = df[df['userId'] == member]
        print('qua')
        setUser = set(user['movieId'].tolist())
        member_list.append(setUser)

    interction = reduce(lambda x, y: x.intersection(y), member_list)
    result = setTot - interction
    chunk_size = len(result) // multiprocessing.cpu_count()
    additional = len(result) % multiprocessing.cpu_count()
    chunks = [list(result)[i:i + chunk_size] for i in range(0, len(result), chunk_size)]

    for i in range(additional):
        index = chunk_size * len(chunks) + i
        if index < len(result):
            chunks[i % len(chunks)].append(list(result)[index])

    pool = multiprocessing.Pool()
    results = pool.map(calculate_prediction, [(group, movie) for movie in chunks])
    pool.close()
    pool.join()

    avg_results = []
    least_results = []
    for res in results:
        avg_results.extend(res[1])
        least_results.extend(res[0])

    avg_results.sort(key=lambda x: x[1], reverse=True)
    least_results.sort(key=lambda x: x[1], reverse=True)

    return [avg_results[:10], least_results[:10]]

def main():
    group = [1, 4, 9]
    top_predictions = checkPredictionPar(group)
    print(top_predictions)


def runMethods(group, movie):
    print(averageMethod(group, movie))
    print(leastMethod(group, movie))
    print(weightedAverageMethod(group, movie))

# Params: group, movie id
runMethods([1, 9, 38], 223)

# if __name__ == "__main__":
#     main()


