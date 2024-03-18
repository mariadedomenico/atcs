import pandas as pd
import numpy as np
import multiprocessing

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

def checkSimilarity(userId) :
    df = pd.read_csv('ml-latest-small/ratings.csv')
    user1 = df[df['userId'] == userId]
    df = df.drop(df[df['userId'] == userId].index)
    prima_riga = df.iloc[0]
    userIndex = prima_riga['userId']
    nuove_righe = []
    top_users = []
    for _, riga in df.iterrows():
        print(userIndex)
        if riga['userId'] == userIndex:
            nuove_righe.append(riga)
        else :
            user2 = pd.DataFrame(nuove_righe)
            similarity = similarityPearson(user1, user2)
            if len(top_users) < 10:
                top_users.append((userIndex, similarity))
            else:
                val_min = min(top_users, key=lambda x: x[1])[1]
                if(val_min < similarity):
                    indice_minimo = top_users.index((min(top_users, key=lambda x: x[1])))
                    top_users[indice_minimo] = (userIndex, similarity)

            userIndex = riga['userId']
            nuove_righe = [riga]

    return top_users
    
def checkPrediction(userId) :

    df = pd.read_csv('ml-latest-small/ratings.csv')
    user1 = df[df['userId'] == userId]
    setTot = set(df['movieId'].tolist())
    setUser1 = set(user1['movieId'].tolist()) 
    risultato = setTot - setUser1

    preds = []
    for movie in risultato :
        pred = prediction(df, user1, movie)
        preds.append(pred)

    preds.sort(reverse=True)
    return preds[:10]

def calculate_prediction(args):
    df, user1, movies = args
    print(movies)
    predictions = []
    for movie in movies:
        print(movie)
        pred = prediction(df, user1, movie)
        predictions.append((movie, pred))
    return predictions


def checkPredictionPar(userId):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    user1 = df[df['userId'] == userId]
    setTot = set(df['movieId'].tolist())
    setUser1 = set(user1['movieId'].tolist()) 
    risultato = setTot - setUser1

    chunk_size = len(risultato) // multiprocessing.cpu_count()
    additional = len(risultato) % multiprocessing.cpu_count()
    chunks = [list(risultato)[i:i + chunk_size] for i in range(0, len(risultato), chunk_size)]

    for i in range(additional):
        index = chunk_size * len(chunks) + i
        if index < len(risultato):
            chunks[i % len(chunks)].append(list(risultato)[index])

    pool = multiprocessing.Pool()
    results = pool.map(calculate_prediction, [(df, user1, movie) for movie in chunks])
    pool.close()
    pool.join()

    merged_results = []
    for result in results:
        merged_results.extend(result)

    merged_results.sort(key=lambda x: x[1], reverse=True)
    return [movie for movie, pred in merged_results[:10]]

def main():
    userId = 1
    top_predictions = checkPredictionPar(userId)
    print("Top 10 predizioni per l'utente con ID", userId)
    for i, movie_id in enumerate(top_predictions, 1):
        print(f"{i}. Movie ID: {movie_id}")

def jaccardCustom(set1, set2) :

    intersection = []
    union = []
    set2result = None
    otherSet = None

    if len(set1) > len(set2) :
        set2result = set2
        otherSet = set1
    else:
        set2result = set1
        otherSet = set2

    for (movie, rating) in set2result :
        presente = any(movie1 == movie and rating - 2 <= rating1 <= rating + 2 for movie1, rating1 in otherSet)
        notIncluded = any(movie1 == movie for movie1, rating1 in otherSet)
        if presente and movie not in intersection:
            intersection.append(movie)
        if notIncluded and movie not in union:
                union.append(movie)

    if(len(union) == 0): return 0
    return len(intersection)/len(union)

def jaccardReal(user1, user2):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    movies1 = set(df.loc[df['userId'] == user1, 'movieId'].tolist())
    movies2 = set(df.loc[df['userId'] == user2, 'movieId'].tolist())
    intersection = movies1.intersection(movies2)
    union = movies1.union(movies2)
    return len(intersection)/len(union)

def runSim(userId1, userId2):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    setUser1 = set(zip(df.loc[df['userId'] == userId1, 'movieId'], df.loc[df['userId'] == userId1, 'rating']))
    setUser2 = set(zip(df.loc[df['userId'] == userId2, 'movieId'], df.loc[df['userId'] == userId2, 'rating']))
    print(jaccardCustom(setUser1, setUser2))

    df1 = df[df['userId'] == userId1]
    df2 = df[df['userId'] == userId2]
    print(similarityPearson(df1, df2))

    print(jaccardReal(userId1, userId2))

runSim(1, 21)

# if __name__ == "__main__":
#     main()




