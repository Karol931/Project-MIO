import string
import pandas as pd
import numpy as np
import spacy
from tweet_cleanup import to_lower, remove_links, remove_punctuation

nlp = spacy.load('en_core_web_md')

def cosine(v1, v2):
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0.0

def sentvec(s):
    sent = nlp(s)

    return np.mean([w.vector for w in sent], axis=0)

# Changing tweets content to vectors
def tweets_to_vectors(tweets_content):
    tweets_vector = []
    
    for tweet_content in tweets_content:
        tweets_vector.append(sentvec(tweet_content))
    
    return tweets_vector

# Printing tweets content and similarities
def print_tweets_and_similarity(t_content_one, t_content_two, t_vector_one, t_vector_two):
    
    print(t_content_one)
    print(t_content_two)
    print(str(cosine(t_vector_one, t_vector_two)) + "\n")

if __name__ == "__main__":

    df = pd.read_csv("trumptweets.csv")

    tweets_content = df['content']
    test_tweets = tweets_content[0:10]
    
    tweets_tweets = to_lower(tweets_content)
    tweets_tweets = remove_links(tweets_tweets)
    tweets_tweets = remove_punctuation(tweets_tweets)

    tweets_vector = tweets_to_vectors(tweets_tweets)

    for i in range(1,len(tweets_vector)):
        print_tweets_and_similarity(tweets_tweets[i-1], tweets_tweets[i] ,tweets_vector[i-1], tweets_vector[i])

    
