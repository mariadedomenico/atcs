import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce

from main import getTopKMovies, getTopKUsers, prediction, similarityPearson
  
def countDisagreements(df, group, movie, pred_dict, n_dict):
    # df = pd.read_csv('ml-latest-small/ratings.csv')
    group_size = len(group)
    sum_ratings = 0
    sum_tot = 0
    ratings = {}
    disagreements_dict = {}
    for member in group:
        movie_pred = pred_dict[member].get(movie, 0)
        if movie_pred == 0:
            user = df[df['userId'] == member]
            neighbor = n_dict[member]
            movie_pred = prediction(df, user, movie, neighbor[movie])
            # ratings[member] = movie_pred
            # sum_ratings += (1/group_size) * movie_pred
        
        ratings[member] = movie_pred
        sum_ratings += (1/group_size) * movie_pred



    for user in group:
        diff = abs(ratings[user] - sum_ratings)
        disagreements_dict[user] = diff

    return (ratings, disagreements_dict)

def weightedAverageMethod(df, group, movie, pred_dict, n_dict):
    
    elem = countDisagreements(df, group, movie, pred_dict, n_dict)
    dis_dict = elem[1]
    ratings = elem[0]
    weighted_sum = 0
    sum_dis = 0
    for user in group:
        rating = ratings[user]
        dis = dis_dict[user]
        weighted_sum += dis*rating
        sum_dis += dis

    if sum_dis == 0:
        return round(sum(ratings.values())/len(group), 2)
    else:
        return round(weighted_sum/sum_dis,2)

def averageMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    for user in group:
        u = df[df['userId'] == user]
        sim_dict = getTopKUsers(user, 40)
        rating = prediction(df, u, movie, sim_dict)
        ratings.append(round(rating,2))
    ratings_sum = sum(ratings)
    pred = ratings_sum / len(ratings)
    return round(pred,2)

def leastMethod(group, movie):
    df = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = []
    for user in group:
        u = df[df['userId'] == user]
        sim_dict = getTopKUsers(user, 40)
        rating = prediction(df, u, movie, sim_dict)
        print(rating)
        ratings.append(round(rating,2))
    return round(min(ratings), 2)

# get average prediction and least prediction all at once
def avgAndLeastMethod(df, group, movie, pred_dict, n_dict):
    ratings = []
    for member in group:
        movie_pred = pred_dict[member].get(movie, 0)
        if movie_pred == 0:
            user = df[df['userId'] == member]
            neighbor = n_dict[member]
            movie_pred = prediction(df, user, movie, neighbor[movie])
        ratings.append(round(movie_pred,2))
    ratings_sum = sum(ratings)
    least = min(ratings)
    avg = ratings_sum / len(ratings)
    return (round(avg,2), round(least, 2))

# get k neighbors that have rated the specific movies 
def getNeighborsPerMovie(movies, n_dict, topUsers, df, k):

    new_dict = {}
    for member in n_dict.keys():
        neighbor = {}
        for movie in movies:
            member_dict = n_dict[member]
            if movie in member_dict.keys():
                neighbor[movie] = member_dict[movie]
            else:
                i=0
                sim_dict = {}
                for id, sim in topUsers:
                    if ((df['userId'] == id) & (df['movieId'] == movie)).any() and i<k:
                        sim_dict[id]=sim
                        i+=1
                    elif i>k:
                        break
                neighbor[movie] = sim_dict
        new_dict[member] = neighbor
    return new_dict

def getTopKMoviesGroup(group, df, k, n) :
    movies_list = set()
    pred_dict = {}
    n_dict = {}

    movies_member = None
    for member in group:
        movies_dict = {}
        movies_member = getTopKMovies(df, member, k, n, similarityPearson)

        for (movie, pred) in movies_member[0]:
            movies_dict[movie] = pred

        pred_dict[member] = movies_dict
        n_dict[member] = movies_member[1]
        movies_list.update([x[0] for x in movies_member[0]])

    predLeast_list = []
    predAvg_list = []

    neighbors = getNeighborsPerMovie(movies_list, n_dict, movies_member[2], df, n)

    for movie in movies_list:
        print(movie)
        (predLeast, predAvg) = avgAndLeastMethod(df, group, movie, pred_dict, neighbors)
        predLeast_list.append((movie, predLeast))
        predAvg_list.append((movie, predAvg))

    predLeast_list.sort(key=lambda x: x[1], reverse=True)
    predAvg_list.sort(key=lambda x: x[1], reverse=True)

    return (predLeast_list[:k], predAvg_list[:k])

def runMethods(group, movie):
    print(averageMethod(group, movie))
    print(leastMethod(group, movie))
    print(weightedAverageMethod(group, movie))
    print(countDisagreements(group, movie))

# Params: group, movie id
# Uncomment this to get group prediction for specific item with different approches
# runMethods([1, 18, 23], 10)

# Params: group, dataset, k-movies, n-neighbors
# Uncomment this to get top k movies for group of three members
# df = pd.read_csv('ml-latest-small/ratings.csv')
# print(getTopKMoviesGroup([1, 4, 9], df, 10, 40))


    

