import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce

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

def createDict(df, df1):
    print('qui')
    df = df.drop(df[df.isin(df1.to_dict('list')).all(axis=1)].index)
    sim_dict = {}
    prima_riga = df.iloc[0]
    userIndex = prima_riga['userId']
    nuove_righe = []
    for _, riga in df.iterrows():
        if riga['userId'] == userIndex:
            nuove_righe.append(riga)
        else :
            user2 = pd.DataFrame(nuove_righe)
            similarity = similarityPearson(df1, user2)
            sim_dict[userIndex] = (user2, similarity)
            userIndex = riga['userId']

    return sim_dict

def prediction(df, df1, movieId) :

    df_elem = df[df['movieId'] == movieId]
    if movieId in df1['movieId'].tolist():
        return df.loc[df['movieId'] == movieId, 'rating'].values[0]
    
    media_ratings1 = df1['rating'].mean()
    num = 0
    similaritySum = 0
    i = 0

    for _, riga in df_elem.iterrows(): 
        df2 = df[df['userId'] == riga['userId']]
        sim = similarityPearson(df1, df2)
        if sim>0 and i<40:
            media_ratings2 = df2['rating'].mean()
            num = num + (sim * (riga['rating'] - media_ratings2))
            similaritySum = similaritySum + sim
        elif i >= 40:
            break

    if(similaritySum != 0):
        prediction = media_ratings1 + (num/similaritySum)
    else:
        prediction = media_ratings1
    return prediction


def weightedAverageMethod(group, movie):
    
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = set()
    # Generate predictions for each user in the group and sum up the scores
    for user in group:
        user = df[df['userId'] == user]
        rating = prediction(df, user, movie)
        print(round(rating,1))
        ratings.add(round(rating,1))
    ratings_sum = sum(ratings)
    # Calculate the average prediction
    pred = ratings_sum / len(ratings)
    return round(pred,1)

def averageMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    # Generate predictions for each user in the group and sum up the scores
    for user in group:
        user = df[df['userId'] == user]
        rating = prediction(df, user, movie)
        print(rating)
        ratings.append(round(rating,2))
    ratings_sum = sum(ratings)
    # Calculate the average prediction
    pred = ratings_sum / len(ratings)
    return round(pred,2)

def leastMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    # Generate predictions for each user in the group and sum up the scores
    for user in group:
        user = df[df['userId'] == user]
        rating = prediction(df, user, movie)
        print(rating)
        ratings.append(round(rating,2))
    return round(min(ratings), 2)

def prediction2(df, df1, movieId, sim_dict) :

    df_elem = df[df['movieId'] == movieId]
    if movieId in df1['movieId'].tolist():
        return df.loc[df['movieId'] == movieId, 'rating'].values[0]
    
    media_ratings1 = df1['rating'].mean()
    num = 0
    similaritySum = 0
    i = 0

    for _, riga in df_elem.iterrows():
        elem = sim_dict[riga['userId']] 
        df2 = elem[0]
        sim = elem[1]
        if sim>0 and i<40:
            media_ratings2 = df2['rating'].mean()
            num = num + (sim * (riga['rating'] - media_ratings2))
            similaritySum = similaritySum + sim
        elif i >= 40:
            break

    if(similaritySum != 0):
        prediction = media_ratings1 + (num/similaritySum)
    else:
        prediction = media_ratings1
    return prediction

def avgAndLeastMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    # Generate predictions for each user in the group and sum up the scores
    for member in group:
        user = df[df['userId'] == member]
        rating = prediction(df, user, movie)
        ratings.append(round(rating,2))
    ratings_sum = sum(ratings)
    least = min(ratings)
    # Calculate the average prediction
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

    intersezione = reduce(lambda x, y: x.intersection(y), member_list)
    risultato = setTot - intersezione
    chunk_size = len(risultato) // multiprocessing.cpu_count()
    additional = len(risultato) % multiprocessing.cpu_count()
    chunks = [list(risultato)[i:i + chunk_size] for i in range(0, len(risultato), chunk_size)]

    for i in range(additional):
        index = chunk_size * len(chunks) + i
        if index < len(risultato):
            chunks[i % len(chunks)].append(list(risultato)[index])

    pool = multiprocessing.Pool()
    results = pool.map(calculate_prediction, [(group, movie) for movie in chunks])
    pool.close()
    pool.join()

    avg_results = []
    least_results = []
    for result in results:
        avg_results.extend(result[1])
        least_results.extend(result[0])

    avg_results.sort(key=lambda x: x[1], reverse=True)
    least_results.sort(key=lambda x: x[1], reverse=True)

    return [avg_results[:10], least_results[:10]]

def main():
    group = [1, 4, 9]
    top_predictions = checkPredictionPar(group)
    print(top_predictions)

if __name__ == "__main__":
    main()


# df = pd.read_csv('ml-latest-small/ratings.csv')
# {1:createSimDict(df, df[df['userId'] == 1]), 4:createSimDict(df, df[df['userId'] == 4]), 9:createSimDict(df, df[df['userId'] == 9])}
# print(leastMethod([1, 4, 9], 50))
    

# def averageMethod2(group, movie, sim_dict):
#     df = pd.read_csv('ml-latest-small/ratings.csv')
#     ratings = []
#     # Generate predictions for each user in the group and sum up the scores
#     for user in group:
#         elem = sim_dict[user]
#         user = elem[0]
#         rating = prediction2(df, user, movie, sim_dict)
#         ratings.append(round(rating,2))
#     ratings_sum = sum(ratings)
#     # Calculate the average prediction
#     pred = ratings_sum / len(ratings)
#     return round(pred,2)

# def leastMethod2(group, movie, sim_dict):
#     df = pd.read_csv('ml-latest-small/ratings.csv')
#     ratings = []
#     # Generate predictions for each user in the group and sum up the scores
#     for user in group:
#         elem = sim_dict[user]
#         user = elem[0]
#         rating = prediction2(df, user, movie, sim_dict)
#         ratings.append(round(rating,2))
#     return round(min(ratings), 2)




