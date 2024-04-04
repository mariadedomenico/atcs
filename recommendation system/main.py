import multiprocessing
import pandas as pd
import numpy as np
import multiprocessing
### TO RUN TOP 10 MOVIES, UNCOMMENT IF NAME == MAIN AND COMMENT RUNSIM. CHANGE USER IN MAIN###
### TO RUN DIFFERENT METHODS, COMMENT IF NAME == MAIN AND UNCOMMENT RUNSIM. CHANGE USER BY PARAM ###

def similarityPearson(df, df1, df2) :

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
        firstElem = (rating1 - media_ratings1)
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


def prediction(df, df1, movieId, neighbors, type=None) :

    df_elem = df[df['movieId'] == movieId]
    if movieId in df1['movieId'].tolist() and type!= 'test':
        return df.loc[df['movieId'] == movieId, 'rating'].values[0]
    
    avg_ratings1 = df1['rating'].mean()
    num = 0
    similaritySum = 0
    i = 0

    for _, row in df_elem.iterrows():
        if row['userId'] in neighbors.keys():
            id = row['userId']
            df2 = df[df['userId'] == id]
            sim = neighbors[id]
            avg_ratings2 = df2['rating'].mean()
            num = num + (sim * (row['rating'] - avg_ratings2))
            similaritySum = similaritySum + abs(sim)

    if(similaritySum != 0):
        prediction = avg_ratings1 + (num/similaritySum)
    else:
        prediction = avg_ratings1
    return prediction


def getTopKUsers(userId, k, df, method, movie = None) :

    user1 = df[df['userId'] == userId]
    df = df.drop(df[df['userId'] == userId].index)

    first_line = df.iloc[0]
    userIndex = first_line['userId']

    new_rows = []
    top_users = []

    for _, row in df.iterrows():
        if row['userId'] == userIndex:
            new_rows.append(row)
        else :
            user2 = pd.DataFrame(new_rows)
            if movie == None or movie in user2['movieId'].tolist():
                similarity = method(df, user1, user2)
                top_users.append((userIndex, similarity))
            userIndex = row['userId']
            new_rows = [row]

    top_users.sort(key=lambda x: abs(x[1]), reverse=True)

    if movie != None :
        sim_dict = {}
        for id, sim in top_users[:k] :
            sim_dict[id] = sim
        return sim_dict

    return top_users[:k]

def makeMovieDict(df, topUsers, movies, k) :

    movies_dict = {}

    for movie in movies:
        i=0
        neighbor_list = {}
        for id, sim in topUsers:
            if ((df['userId'] == id) & (df['movieId'] == movie)).any() and i<k:
                neighbor_list[id]=sim
                i+=1
            elif i>k:
                break
        movies_dict[movie] = neighbor_list

    return movies_dict
    
def getTopKMovies(df, userId, k, n, method=similarityPearson) :

    # df = pd.read_csv('ml-latest-small/ratings.csv')

    topUsers = getTopKUsers(userId, len(df['userId']), df, method)
    movies_other_users = set()
    for user, sim in topUsers[:10] :
        u = df[df['userId'] == user]
        movies_other_users.update(u['movieId'].tolist())

    user1 = df[df['userId'] == userId]
    user1_movies = set(user1['movieId'].tolist()) 
    result = movies_other_users - user1_movies
    # print(len(result))

    movie_dict = makeMovieDict(df, topUsers, result, n)

    preds = []
    for movie in movie_dict.keys() :
        # print(movie)
        pred = prediction(df, user1, movie, movie_dict[movie])
        preds.append((movie, pred))
        
    preds.sort(key=lambda x: x[1], reverse=True)
    
    return (preds[:k], movie_dict, topUsers)

def jaccardCustom(df, df1, df2) :

    userId1 = df1['userId'].values[0]
    userId2 = df2['userId'].values[0]
    set1 = set(zip(df.loc[df['userId'] == userId1, 'movieId'], df.loc[df['userId'] == userId1, 'rating']))
    set2 = set(zip(df.loc[df['userId'] == userId2, 'movieId'], df.loc[df['userId'] == userId2, 'rating']))

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
        included = any(movie1 == movie and rating - 2 <= rating1 <= rating + 2 for movie1, rating1 in otherSet)
        notIncluded = any(movie1 == movie for movie1, rating1 in otherSet)
        if included and movie not in intersection:
            intersection.append(movie)
        if notIncluded and movie not in union:
            union.append(movie)

    if(len(union) == 0): return 0
    return len(intersection)/len(union)

def jaccardReal(df, df1, df2):
    user1 = df1['userId'].values[0]
    user2 = df2['userId'].values[0]

    movies1 = set(df.loc[df['userId'] == user1, 'movieId'].tolist())
    movies2 = set(df.loc[df['userId'] == user2, 'movieId'].tolist())
    intersection = movies1.intersection(movies2)
    union = movies1.union(movies2)
    return len(intersection)/len(union)

def cosineSimilarity(df, df1, df2):

    ratings1= (df1['rating'] ** 2).sum()
    ratings2= (df2['rating'] ** 2).sum()
    denominator = np.sqrt(ratings1)*np.sqrt(ratings2)

    movie_set1 = set(df1['movieId'].tolist())
    movie_set2 = set(df2['movieId'].tolist())

    movie_intersection = movie_set1.intersection(movie_set2)
    
    numerator = 0
    for movie in movie_intersection :
        rating1 = df1.loc[df1['movieId'] == movie, 'rating'].values[0]
        rating2 = df2.loc[df2['movieId'] == movie, 'rating'].values[0]
        numerator = numerator + (rating1 * rating2)

    if denominator == 0 :
        return 0
    else:
        similarity = numerator/denominator
        return similarity

def improvedTraingleSimilarity(df, df1, df2):

    mean_rating1 = df1['rating'].mean()
    mean_rating2 = df1['rating'].mean()

    merged_df = pd.merge(df1, df2, on='movieId', suffixes=('_df1', '_df2'), how='inner')
    merged_df['diff'] = merged_df['rating_df1'] - merged_df['rating_df2']
    sum_of_squared_differences = (merged_df['diff'] ** 2).sum()
    sum_of_squared_items_user1 = (merged_df['rating_df1'] ** 2).sum()
    sum_of_squared_items_user2 = (merged_df['rating_df2'] ** 2).sum()

    denominator = (np.sqrt(sum_of_squared_items_user1) + np.sqrt(sum_of_squared_items_user2))
    sim_triangle = 0
    if(denominator != 0):
        sim_triangle = 1 - (np.sqrt(sum_of_squared_differences)/denominator)

    df1.loc[:, 'rating'] = df1['rating'] - mean_rating1
    standardDev1 = np.sqrt(((df1['rating'] ** 2).sum())/len(df1['rating']))

    df2.loc[:, 'rating'] = df2['rating'] - mean_rating2
    standardDev2 = np.sqrt(((df2['rating'] ** 2).sum())/len(df2['rating']))

    sim_urp = 1 - 1/(1 + np.exp(-(abs(mean_rating1-mean_rating2)*abs(standardDev1-standardDev2))))

    return sim_triangle * sim_urp


def runSim(userId1, userId2):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    df1 = df[df['userId'] == userId1]
    df2 = df[df['userId'] == userId2]

    print(round(jaccardCustom(df, df1, df2), 2))
    print(round(similarityPearson(df, df1, df2), 2))
    print(round(jaccardReal(df, df1, df2), 2))
    print(round(improvedTraingleSimilarity(df, df1, df2), 2))
    print(round(cosineSimilarity(df, df1, df2), 2))


# Params: user id, movie id
#runSim(1, 2)
# if __name__ == '__main__':
# df = pd.read_csv('ml-latest-small/ratings.csv')

# res = getTopKMovies(df, 1, 10, 40)
# print(res[0])








