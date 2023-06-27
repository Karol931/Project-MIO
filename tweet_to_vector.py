import string
import pandas as pd
import os
import numpy as np
import spacy
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from tweet_cleanup import trim_tweets, delete_empty_tweets

# nlp = spacy.load('en_core_web_md')

# def cosine(v1, v2):
#     if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
#         return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     else:
#         return 0.0

# def sentvec(s):
#     sent = nlp(s)

#     return np.mean([w.vector for w in sent], axis=0)

# # Changing tweets content to vectors
# def tweets_to_vectors(tweets_content):
#     tweets_vector = []
    
#     for tweet_content in tweets_content:
#         tweets_vector.append(sentvec(tweet_content))
    
#     return tweets_vector

# # Printing tweets content and similarities
# def print_tweets_and_similarity(t_content_one, t_content_two, t_vector_one, t_vector_two):
    
#     print(t_content_one)
#     print(t_content_two)
#     print(str(cosine(t_vector_one, t_vector_two)) + "\n")

# Makes file with tweets as rows
def prepare_training_data(df):
    df.to_csv("tweet_words.txt", columns=["content"], header=None, index=False)

# Trains model from file with tweets as rows
def train_model(path):
    model = fasttext.train_unsupervised(path, model='skipgram', lr=0.1, lrUpdateRate=100, dim=100, loss='hs')

    return model

# Converts tweet into vector of length 100
def get_tweets_vector_list(tweets_content, model):

    tweets_vectors = [model.get_sentence_vector(tweet) for tweet in tweets_content]


    return tweets_vectors



if __name__ == "__main__":

    df = pd.read_csv("trumptweets.csv")

    df["content"] = trim_tweets(df["content"])

    df = delete_empty_tweets(df)

    prepare_training_data(df)

    model = train_model('./tweet_words.txt')
    tweets_content = df["content"]

    model.save_model('saved_model_trumptweets.bin')

    tweets_vectors = get_tweets_vector_list(tweets_content, model)

    print(len(tweets_vectors[0]))

    # for i in range(10):
    #     tweet1 = tweets_content[i]
    #     tweet2 = tweets_content[i+1]

    #     vector1 = model.get_sentence_vector(tweet1)
    #     vector2 = model.get_sentence_vector(tweet2)

    #     similarity = cosine_similarity(np.array(vector1).reshape(1, -1), np.array(vector2).reshape(1, -1))
    #     print(tweet1)
    #     print(tweet2)
    #     print(similarity)
    #     print()

    # print(df["content"])

    
