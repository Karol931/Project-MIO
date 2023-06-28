import pandas as pd
import numpy as np
import re
import string


# Removes links from tweet
def remove_links(tweets_content):
    tweets_content = tweets_content.replace(to_replace="https?://\S+", value=" ", regex=True)
    return tweets_content


def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)

    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)

    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    return text

# Makes tweet lowercase
def to_lower(tweets_content):
    tweets_content = tweets_content.str.lower()

    return tweets_content


# Removes punctuation from tweet
def remove_punctuation(tweets_content):
    tweets_content = tweets_content.replace(to_replace="[^A-Za-z' ]", value=" ", regex=True)

    return tweets_content


def remove_tags(tweets_content):
    tweets_content = tweets_content.replace(to_replace="(@)\s?(\w+)", value=" ", regex=True)

    return tweets_content


def remove_hashtags(tweets_content):
    tweets_content = tweets_content.replace(to_replace="(#)\s?(\w+)", value=" ", regex=True)

    return tweets_content


def remove_many_whitespaces(tweets_content):
    tweets_content = tweets_content.replace(to_replace="\s{2,}", value=" ", regex=True)

    return tweets_content


def remove_pictures(tweets_content):
    tweets_content = tweets_content.replace(to_replace="pic\.twitter\.com/\S+", value=" ", regex=True)

    return tweets_content


def trim_tweets(tweets_content):
    tweets_content = remove_pictures(tweets_content)
    tweets_content = remove_links(tweets_content)
    tweets_content = remove_hashtags(tweets_content)
    tweets_content = remove_tags(tweets_content)
    tweets_content = remove_punctuation(tweets_content)
    tweets_content = remove_many_whitespaces(tweets_content)
    tweets_content = to_lower(tweets_content)
    tweets_content = tweets_content.apply(lambda x : decontraction(x))

    return tweets_content

# to rename
def delete_empty_tweets(df):
    for i in df.index:
        if (len(str(df["content"][i]).split()) < 4):
            df = df.drop(index=i)

    return df


if __name__ == "__main__":
    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])

    df = delete_empty_tweets(df)

    print(len(df))
    # print(df["content"].to_numpy())
        