import pandas as pd
import numpy as np
import re
import string

# Removes links from tweet
def remove_links(tweets_content):
   
    tweets_content = tweets_content.replace(to_replace = "https?://\S+", value = " ", regex = True)
    return tweets_content

# Makes tweet lowercase
def to_lower(tweets_content):
    
    tweets_content = tweets_content.str.lower()
    
    return tweets_content

# Removes punctuation from tweet
def remove_punctuation(tweets_content):
    
    tweets_content = tweets_content.replace(to_replace = "[^A-Za-z' ]", value = "", regex = True)
    
    return tweets_content

def remove_tags(tweets_content):

    tweets_content = tweets_content.replace(to_replace = "(@)\s?(\w+)", value = " ", regex = True)

    return tweets_content

def remove_hashtags(tweets_content):

    tweets_content = tweets_content.replace(to_replace = "(#)\s?(\w+)", value = " ", regex = True)

    return tweets_content

def remove_many_whitespaces(tweets_content):

    tweets_content = tweets_content.replace(to_replace = "\s{2,}", value = " ", regex = True)

    return tweets_content

def remove_pictures(tweets_content):

    tweets_content = tweets_content.replace(to_replace = "pic.twitter.com/\S+", value = " ", regex = True)

    return tweets_content

def trim_tweets(tweets_content):
    tweets_content = remove_links(tweets_content)
    tweets_content = remove_hashtags(tweets_content)
    tweets_content = remove_tags(tweets_content)
    tweets_content = remove_pictures(tweets_content)
    tweets_content = remove_punctuation(tweets_content)
    tweets_content = remove_many_whitespaces(tweets_content)
    tweets_content = to_lower(tweets_content)

    return tweets_content

# nie dziala
def delete_empty_tweets(df):

    tweets_content = df["content"]

    for i in range(len(df)):
        if(len(tweets_content[i]) < 2):
            df = df.drop(index = i)
    
    return df


if __name__ == "__main__":
    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])

    df = delete_empty_tweets(df)

    print(len(df))
    # print(df["content"].to_numpy())
        