import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import normalize_quantile, normalize_log, trim_tweets, delete_empty_tweets



def divide_data():
    if not os.path.exists('./tweet_data'):
        os.mkdir("./tweet_data")

    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])
    df = delete_empty_tweets(df)

    data = df[['content', 'retweets']].values
    quantile_rt = normalize_quantile(df['retweets']).values
    log_rt, _ = normalize_log(df['retweets'])
    # print(log_rt)
    data = np.column_stack((data, quantile_rt, log_rt))
    
    # print(data, data.shape)
    # data_train, data_test = split_data(data)
    
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=2142037, stratify=np.ceil(quantile_rt))

    pd.DataFrame(data_train).to_csv("./tweet_data/data_train.csv", header=None, index=False)
    pd.DataFrame(data_test).to_csv("./tweet_data/data_test.csv", header=None, index=False)

def load_tweet_data():
    if not os.path.exists('./tweet_data'):
        divide_data()

    df_train = pd.read_csv("./tweet_data/data_train.csv", header=None)
    df_test = pd.read_csv("./tweet_data/data_test.csv", header=None)

    text_train = df_train[0].values
    retweets_train = df_train[1].values
    quantile_rt_train = df_train[2].values
    log_rt_train = df_train[3].values


    text_test = df_test[0].values
    retweets_test = df_test[1].values
    quantile_rt_test = df_test[2].values
    log_rt_test = df_test[3].values


    return text_train, text_test, retweets_train, retweets_test, quantile_rt_train, quantile_rt_test, log_rt_train, log_rt_test

if __name__ == "__main__":
    divide_data()
