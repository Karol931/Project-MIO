import numpy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tweet_cleanup import trim_tweets, delete_empty_tweets
import fasttext


def divide_data(bins=3, log = True):
    if not os.path.exists('./tweet_data'):
        os.mkdir("./tweet_data")

    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])
    df = delete_empty_tweets(df)


    df.drop(df[df['retweets'] < np.exp(2)].index, inplace=True)
    df.drop(df[df['retweets'] > np.exp(10.5)].index, inplace=True)

    # lower_quartile = df['retweets'].quantile(0.5)
    # upper_quartile = df['retweets'].quantile(0.75)

    df['retweets_log'] = np.log(df['retweets'] + 1)

    if log:
        low_ = min(df['retweets_log'])
        high_ = max(df['retweets_log'])
        lspc = np.exp(np.linspace(low_, high_, bins + 1)) - 1
        lspc[0] = float('-inf')
        lspc[-1] = float('inf')
        print(lspc)
    else:
        lspc = df['retweets'].quantile(np.linspace(0, 1, bins + 1)).values
        lspc[0] = float('-inf')
        lspc[-1] = float('inf')
        print(lspc)


    df['class'] = pd.cut(df['retweets'], bins=lspc,
                         labels=[str(i + 1) for i in range(bins)], right=False)

    df['content'] = "__label__" + df['class'].astype(str) + " " + df['content'].astype(str)


    plt.figure()
    plt.hist([df['retweets_log'][df['class'] == str(i+1)] for i in range(bins)], 100)


    data_train, data_test = train_test_split(df[['content', 'retweets', 'class']], test_size=0.2)


    data_train.to_csv("./tweet_data/data_train.train", columns=['content'], header=None, index=False)
    data_test.to_csv("./tweet_data/data_test.test", columns=['content'], header=None, index=False)

    count_data = {'data_count': df['class'].value_counts(),
                  'data_train_count': data_train['class'].value_counts(),
                    'data_test_count': data_test['class'].value_counts()}
    df_count = pd.DataFrame(count_data)
    print(df_count)


def make_model(bins,log = True):
    divide_data(bins, log)

    model = fasttext.train_supervised("./tweet_data/data_train.train", lr=.05, epoch=15, wordNgrams=2, dim=50)
    print(model.test("./tweet_data/data_train.train"))
    print(model.test("./tweet_data/data_test.test"))
    ms = model.test_label("./tweet_data/data_test.test")
    for k, v in ms.items():
        print(k, v)


if __name__ == "__main__":
    for i in range(3, 6):
        print("----------------------------------------")
        print("nr of bins: ", i)
        make_model(i, False)

    plt.show()
