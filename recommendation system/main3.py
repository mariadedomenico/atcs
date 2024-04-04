import main2, main
import pandas as pd 
import random

def getKMoviesGroup(df, userId, k, n, method, top_neighbors):
    movies_other_users = set()
    for user, sim in top_neighbors[:10] :
        u = df[df['userId'] == user]
        movies_other_users.update(u['movieId'].tolist())

    user1 = df[df['userId'] == userId]
    user1_movies = set(user1['movieId'].tolist()) 
    result = movies_other_users - user1_movies

    movie_dict = main.makeMovieDict(df, top_neighbors, result, n)

    preds = []
    for movie in movie_dict.keys() :
        pred = main.prediction(df, user1, movie, movie_dict[movie])
        preds.append((movie, pred))
        
    preds.sort(key=lambda x: x[1], reverse=True)
    
    return (preds[:k], movie_dict, top_neighbors)

def cleanNeighbor(chunk, list) :
    pairs_included = [(id, pred) for id, pred in list if id in chunk['userId'].values]
    return pairs_included

def getSatisfaction(group, group_preds, user_preds, sat_group) :

    sat = {}
    num = sum(pred[1] for pred in group_preds)
    for user in group:
        den = sum(user_preds[user].values())
        if den == 0:
            sat[user] = 0
            sat_group[user] = sat_group.get(user, []) + [0]
        else:
            sat[user] = num/den
            sat_group[user] = sat_group.get(user, []) + [num/den]

    print(sat)
    print(sat_group)
    return (sat, sat_group)


def satOverall(satisfactions, n_iter):
    return sum(satisfactions)/n_iter

def userDis(satisfactions, sat):
    return max(satisfactions) - sat

def weightSIAA(b, satO, userDis):
    return (1-b)*(1-satO) + b*userDis

def hybridMethod(chunk, group, movie, pred_dict, neighbors, alfa) :

    (avg, least) = main2.avgAndLeastMethod(chunk, group, movie, pred_dict, neighbors)
    return round((1-alfa)*avg + alfa*least, 2)

def weightedAvgMethod(chunk, group, movie, pred_dict, neighbors, alfa) :

    res = (1-alfa)*main2.weightedAverageMethod(chunk, group, movie, pred_dict, neighbors)
    return res

def getKTopMoviesSequential(group, n_iter, k, n, neighbors_group, df, method=hybridMethod):

    total_rows = sum(1 for line in open('ml-latest-small/ratings.csv')) - 1  
    chunk_size = total_rows // n_iter
    df_chunks = pd.read_csv('ml-latest-small/ratings.csv', chunksize = chunk_size)

    pred_dict = {}
    n_dict = {}
    iter_list = []
    sat_group = ({}, {})


    result = pd.DataFrame()
    alfa = 0
    i = 0

    for chunk in df_chunks :
        print("Iteration:", i+1,", alfa =",alfa)
        movies_list = set()
        result = pd.concat([result, chunk])
        movies_member = None
        for member in group:
            movies_dict = {}
            neighbor_chunk = cleanNeighbor(result, neighbors_group[member])
            movies_member = getKMoviesGroup(df, member, k, n, main.similarityPearson, neighbor_chunk)

            for (movie, pred) in movies_member[0]:
                movies_dict[movie] = pred

            pred_dict[member] = movies_dict
            n_dict[member] = movies_member[1]
            movies_list.update([x[0] for x in movies_member[0]])


        neighbors = main2.getNeighborsPerMovie(movies_list, n_dict, movies_member[2], df, n)
        pred_list = []
        for movie in movies_list:
            pred = method(df, group, movie, pred_dict, neighbors, alfa)
            pred_list.append((movie, pred))

        pred_list.sort(key=lambda x: x[1], reverse=True)
        sat_group = getSatisfaction(group, pred_list[:k], pred_dict, sat_group[1])
        sat_list = sat_group[0].values()
        print("Group's satisfaction: ",sum(sat_list)/len(group))
        alfa = max(sat_list) - min(sat_list)

        iter_list.append(pred_list[:k])
        i+=1

    return iter_list

def getPrediction(movie, u, pred_dict, df, result, neighbors):
    pred = pred_dict.get(movie, -1)
    if pred == -1:
        df1 = df[df['userId'] == u]
        neighbor = neighbors[u]
        pred = main.prediction(result, df1, movie, neighbor[movie])
    
    return pred


def siaaMethod(group, n_iter, k, n, neighbors_group, df):

    total_rows = sum(1 for line in open('ml-latest-small/ratings.csv')) - 1  
    chunk_size = total_rows // n_iter
    df_chunks = pd.read_csv('ml-latest-small/ratings.csv', chunksize = chunk_size)

    pred_dict = {}
    n_dict = {}
    iter_list = []
    sat_group = ({}, {})

    result = pd.DataFrame()
    alfa = 0
    i = 0

    for chunk in df_chunks :
        print("Iteration:", i+1,", alfa =",alfa)
        movies_list = set()
        result = pd.concat([result, chunk])
        movies_member = None
        for member in group:
            movies_dict = {}
            neighbor_chunk = cleanNeighbor(result, neighbors_group[member])
            movies_member = getKMoviesGroup(df, member, k, n, main.similarityPearson, neighbor_chunk)

            for (movie, pred) in movies_member[0]:
                movies_dict[movie] = pred

            pred_dict[member] = movies_dict
            n_dict[member] = movies_member[1]
            movies_list.update([x[0] for x in movies_member[0]])


        neighbors = main2.getNeighborsPerMovie(movies_list, n_dict, movies_member[2], result, n)
        pred_list = []
        for movie in movies_list:
            score = 0
            if i == 0:
                score = hybridMethod(df, group, movie, pred_dict, neighbors, alfa)
            else:
                for u in group:
                    sat_user = sat_group[1][u]
                    satO = satOverall(sat_user, n_iter)
                    dis = userDis(sat_list, sat_group[0][u])
                    w = weightSIAA(sat_group[0][u], satO, dis)
                    score = score + w*getPrediction(movie, u, pred_dict[u], df, result, neighbors)
            pred_list.append((movie, round(score, 2)))

        pred_list.sort(key=lambda x: x[1], reverse=True)
        sat_group = getSatisfaction(group, pred_list[:k], pred_dict, sat_group[1])
        sat_list = sat_group[0].values()
        print("Group's satisfaction: ",sum(sat_list)/len(group))
        alfa = max(sat_list) - min(sat_list)

        iter_list.append(pred_list[:k])
        i+=1

    return iter_list

def getNeighborsDict(userId, df, n):

    neighbors_group = {}
    neighbor = main.getTopKUsers(userId, n, df, main.similarityPearson)
    neighbors_group[userId] = neighbor

    return neighbors_group

def getSimilarUsers(k, neighbor):

    similar_users = [(id, sim) for (id, sim) in neighbor if sim > 0.5]
    return random.sample(similar_users, k)

def getDisimilarUsers(k, neighbor):

    disimilar_users = [(id, sim) for (id, sim) in neighbor if sim < -0.5]
    return random.sample(disimilar_users, k)

def makeGroup(userId, sim, dis, user_dict, num):

    group = [(userId, 1)] + getSimilarUsers(sim, user_dict[userId]) + getDisimilarUsers(dis, user_dict[userId])
    for member in group:
        if member not in user_dict.keys():
            user_dict.update(getNeighborsDict(member, df, num))
        
    return (group, user_dict)


df = pd.read_csv('ml-latest-small/ratings.csv') 
user_num =  df['userId'].nunique()

user_dict = getNeighborsDict(1, df, user_num)

(group, n_dict) = makeGroup(1, 2, 0, user_dict, user_num)
print('GRUPPO 1: ' + str(group))
print(siaaMethod([elem[0] for elem in group], 3, 10, 40, n_dict, df))
(group, n_dict) = makeGroup(1, 0, 2, user_dict, user_num)
print('GRUPPO 2: ' + str(group))
print(siaaMethod([elem[0] for elem in group], 3, 10, 40, n_dict, df))
(group, n_dict) = makeGroup(1, 1, 1, user_dict, user_num)
print('GRUPPO 3: ' + str(group))
print(siaaMethod([elem[0] for elem in group], 3, 10, 40, n_dict, df))

# (group, n_dict) = makeGroup(1, 0, 2, user_dict, user_num, [1, 236, 44])
# print('GRUPPO 1: ' + str(group))
# print(getKTopMoviesSequential(group, 3, 10, 40, n_dict, df))
# (group, n_dict) = makeGroup(1, 0, 2, user_dict, user_num, [1, 252, 71])
# print('GRUPPO 2: ' + str(group))
# print(getKTopMoviesSequential(group, 3, 10, 40, n_dict, df))
# (group, n_dict) = makeGroup(1, 1, 1, user_dict, user_num, [1, 479, 71])
# print('GRUPPO 3: ' + str(group))
# print(getKTopMoviesSequential(group, 3, 10, 40, n_dict, df))





