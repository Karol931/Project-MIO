import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from tweet_cleanup import delete_empty_tweets, trim_tweets


def normalize_quantile(data):

    data = data.rank(pct=True)

    return data

def normalize_log(data):

    data = np.log(data + 1)
    data = data/ max(data)

    return data

if __name__ == "__main__":

    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])

    df = delete_empty_tweets(df)
    retweets = df["retweets"]
    favourites = df["favorites"]

    log_retweets = normalize_log(retweets)

    quant_retweets = normalize_quantile(retweets)

    # plt.hist(log_retweets)
    # plt.figure()
    # plt.hist(quant_retweets)
    # plt.figure()
    # plt.hist(retweets)
    # plt.show()


    # for i in range(100):
    #     print(str(log_retweets[i]) + " " + str(quant_retweets[i]) + " " + str(retweets[i]))