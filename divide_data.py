import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tweet_cleanup import trim_tweets, delete_empty_tweets
from normalize_tweet_popularity import normalize_quantile


def divide_data():
    if not os.path.exists('./tweet_data'):
        os.mkdir("./tweet_data")

    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])
    df = delete_empty_tweets(df)

    data = df[['content', 'retweets']].values

    quantile_rt = normalize_quantile(df['retweets']).values
    data = np.column_stack((data, quantile_rt))

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=2142037)

    pd.DataFrame(data_train).to_csv("./tweet_data/data_train.csv", header=None, index=False)
    pd.DataFrame(data_test).to_csv("./tweet_data/data_test.csv", header=None, index=False)

def load_tweet_data():
    if not os.path.exists('./tweet_data'):
        divide_data()

    df_train = pd.read_csv("./tweet_data/data_train.csv", header=None)
    df_test = pd.read_csv("./tweet_data/data_test.csv", header=None)

    text_train = df_train[0].values
    retweets_train = df_train[1].values
    normalized_rt_train = df_train[2].values

    text_test = df_test[0].values
    retweets_test = df_test[1].values
    normalized_rt_test = df_test[2].values

    return text_train, text_test, retweets_train, retweets_test, normalized_rt_train, normalized_rt_test

if __name__ == "__main__":
    divide_data()
