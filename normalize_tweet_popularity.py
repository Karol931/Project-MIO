import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from tweet_cleanup import delete_empty_tweets, trim_tweets


def normalize_quantile(data):

    data = data.rank(pct=True)*100
    
    return data

def normalize_log(data):

    data = np.log(data + 1)
    data = data/ max(data)

    return data

def get_tweet_score(favourite, retweets, scale = 0.5):

    score = favourite * scale + retweets * scale

    return score

def denormalize_quantile(data, value):

    if(value > 100): 
        value = 100
    elif(value < 0):
        value = 0
    # print(data.quantile(value/100))
    return data.quantile(value/100)


if __name__ == "__main__":

    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])

    df = delete_empty_tweets(df)
    retweets = df["retweets"]
    favourites = df["favorites"]

    log_retweets = normalize_log(retweets)

    quant_retweets = normalize_quantile(retweets)

    log_favourites = normalize_log(favourites)

    quant_favourites = normalize_quantile(favourites)

    quant_score = get_tweet_score(quant_favourites, quant_retweets)

    

    # plt.hist(log_retweets)
    # plt.figure()
    # plt.hist(quant_retweets)
    # plt.figure()
    # plt.hist(retweets)
    # plt.show()


    for i in range(100):
        print(str(quant_retweets[i]) + " " + str(denormalize_quantile(retweets, quant_retweets[i])) + " " + str(retweets[i]))