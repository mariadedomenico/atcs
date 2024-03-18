import pandas as pd
import numpy as np
import multiprocessing

### TO RUN TOP 10 MOVIES, UNCOMMENT IF NAME == MAIN AND COMMENT RUNSIM. CHANGE USER IN MAIN###
### TO RUN DIFFERENT METHODS, COMMENT IF NAME == MAIN AND UNCOMMENT RUNSIM. CHANGE USER BY PARAM ###

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

def checkSimilarity(userId) :
    df = pd.read_csv('ml-latest-small/ratings.csv')
    user1 = df[df['userId'] == userId]
    df = df.drop(df[df['userId'] == userId].index)
    first_line = df.iloc[0]
    userIndex = first_line['userId']
    new_rows = []
    top_users = []
    for _, row in df.iterrows():
        print(userIndex)
        if row['userId'] == userIndex:
            new_rows.append(row)
        else :
            user2 = pd.DataFrame(new_rows)
            similarity = similarityPearson(user1, user2)
            top_users.append((userIndex, similarity))
            userIndex = row['userId']
            new_rows = [row]

    return top_users
    
def checkPrediction(userId) :

    df = pd.read_csv('ml-latest-small/ratings.csv')
    user1 = df[df['userId'] == userId]
    setTot = set(df['movieId'].tolist())
    setUser1 = set(user1['movieId'].tolist()) 
    result = setTot - setUser1

    preds = []
    for movie in result :
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
    result = setTot - setUser1

    chunk_size = len(result) // multiprocessing.cpu_count()
    additional = len(result) % multiprocessing.cpu_count()
    chunks = [list(result)[i:i + chunk_size] for i in range(0, len(result), chunk_size)]

    for i in range(additional):
        index = chunk_size * len(chunks) + i
        if index < len(result):
            chunks[i % len(chunks)].append(list(result)[index])

    pool = multiprocessing.Pool()
    results = pool.map(calculate_prediction, [(df, user1, movie) for movie in chunks])
    pool.close()
    pool.join()

    merged_results = []
    for res in results:
        merged_results.extend(res)

    merged_results.sort(key=lambda x: x[1], reverse=True)
    return [movie for movie, pred in merged_results[:10]]

def main():
    userId = 1
    top_predictions = checkPredictionPar(userId)
    print("Top 10 predictions for user with ID", userId)
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
    print(round(jaccardCustom(setUser1, setUser2), 2))

    df1 = df[df['userId'] == userId1]
    df2 = df[df['userId'] == userId2]
    print(round(similarityPearson(df1, df2), 2))

    print(round(jaccardReal(userId1, userId2), 2))

# Params: user id, movie id
runSim(13, 50)
# if __name__ == "__main__":
#     main()




