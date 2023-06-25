import pandas as pd
import os
import numpy as np
import fasttext
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from tweet_cleanup import trim_tweets, delete_empty_tweets
from normalize_tweet_popularity import normalize_log, denormalize_log, normalize_quantile

saved_model_name = 'saved_model_trumptweets.bin'
model = fasttext.load_model(saved_model_name)

df = pd.read_csv("trumptweets.csv")

print("trimming data")
df["content"] = trim_tweets(df["content"])
df = delete_empty_tweets(df)
print("DONE")


print("mapping vectors")
sentence_vectors = [model.get_sentence_vector(sentence) for sentence in df["content"]]
# sentence_vectors = df.content.map(model.get_sentence_vector)
print("DONE")

retweets = df["retweets"]
quant_retweets = normalize_quantile(retweets)

data_train, data_test, label_train, label_test = train_test_split(sentence_vectors, quant_retweets, test_size=0.2)

saved_model_name = 'saved_model_adam_200_100_50_relu.joblib'
if not os.path.exists(saved_model_name):
    network = MLPRegressor(solver='adam', hidden_layer_sizes=(200, 100, 50), max_iter=15000, tol=10e-6, activation='relu')
    network.fit(data_train, label_train)
    joblib.dump(network, saved_model_name)
else:
    network = joblib.load(saved_model_name)

label_pred = network.predict(data_test)

mse = mean_squared_error(label_test, label_pred)
print("Mean Squared Error:", mse)

test = df["content"][3548]

test_rt = quant_retweets[3548]

print(test, test_rt)
test_pred = network.predict(model.get_sentence_vector(test).reshape(1, -1))

print(test_pred)

test = df["content"][548]

test_rt = quant_retweets[548]

print(test, test_rt)
test_pred = network.predict(model.get_sentence_vector(test).reshape(1, -1))

print(test_pred)