import pandas as pd
import numpy as np
import re
import string

# Removes links from tweet
def remove_links(tweets_content):
   
    tweets_content = tweets_content.replace(to_replace = "http://\S+", value = "", regex = True)
    
    return tweets_content

# Makes tweet lowercase
def to_lower(tweets_content):
    
    tweets_content = tweets_content.str.lower()
    
    return tweets_content

# Removes punctuation from tweet
def remove_punctuation(tweets_content):
    
    tweets_content = tweets_content.replace(to_replace = "[^\w\s]", value = "", regex = True)
    
    return tweets_content


if __name__ == "__main__":
    df = pd.read_csv("trumptweets.csv")

    tweets_content = df['content']

    test = tweets_content[0:10]
    print(test.to_numpy())
    test = to_lower(test)
    test = remove_links(test)
    test = remove_punctuation(test)
    print(test.to_numpy())


        