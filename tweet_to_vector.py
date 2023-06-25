import pandas as pd
import numpy as np
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from tweet_cleanup import trim_tweets, delete_empty_tweets


# Makes file with tweets as rows
def prepare_training_data(df):
    df.to_csv("tweet_words.txt", columns=["content"], header=None, index=False)

# Trains model from file with tweets as rows
def train_model(path):
    model = fasttext.train_unsupervised(path, model='skipgram', lr=0.1, lrUpdateRate=100, dim=100, loss='hs')

    return model

# Converts tweet into vector of length 100
def get_tweets_vector_list(tweets_content, model):

    tweets_vectors = []

    for tweet in tweets_content:
        vector = model.get_sentence_vector(tweet)
        tweets_vectors.append(vector)

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

    
