import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fasttext
import joblib

from divide_data import load_tweet_data
from tweet_to_vector import get_tweets_vector_list, train_model, prepare_training_data

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


def load_fasttext_model(df, fasttext_model_name: str, model_size: int):
    if os.path.exists(fasttext_model_name):
        model = fasttext.load_model(fasttext_model_name)
    else:
        df.to_csv("tweet_words.txt", columns=["content"], header=None, index=False)
        model = fasttext.train_unsupervised('./tweet_words.txt', model='skipgram', lr=0.1, lrUpdateRate=100,
                                            dim=model_size, loss='hs')
        model.save_model(fasttext_model_name)
    return model


def make_nlp_name(hidden_layer_sizes: tuple, activation: str, normalize: str):
    out = "model_mlp_"
    for hls in hidden_layer_sizes:
        out += str(hls) + "_"
    out += activation + "_"
    out += normalize
    out += ".joblib"
    return out


def normalize_log(retweets_train, retweets_test):
    label_test = np.log(retweets_test + 1)
    label_train = np.log(retweets_train + 1)
    label_max = max(label_test.max(), label_train.max())
    label_test /= label_max
    label_train /= label_max
    return label_train, label_test, label_max


def origin_log(rt_pred_train, rt_pred_test, label_max):
    pred_train_origin = np.exp(rt_pred_train * label_max) - 1
    pred_test_origin = np.exp(rt_pred_test * label_max) - 1

    return pred_train_origin, pred_test_origin


if __name__ == "__main__":

    text_train, text_test, retweets_train, retweets_test, normalized_rt_train, normalized_rt_test = load_tweet_data()

    retweets = np.concatenate((retweets_train, retweets_test))
    content = np.concatenate((text_train, text_test))

    df = pd.DataFrame(data={'content': content, 'retweets': retweets})

    model_size = 100  #zmienna do wytestowania
    fasttext_model_name = "saved_model_trumptweets_" + str(model_size) + ".bin"
    model = load_fasttext_model(df, fasttext_model_name, model_size)

    vectors_train = get_tweets_vector_list(text_train, model)
    vectors_test = get_tweets_vector_list(text_test, model)

    #zmienne do wytestowania
    hidden_layer_sizes = (52, 24,)
    activation = 'relu'

    normalize = 'log' #dont change

    nlp_model_name = make_nlp_name(hidden_layer_sizes, activation, normalize)
    print(nlp_model_name)

    label_train, label_test, label_max = normalize_log(retweets_train, retweets_test)

    if os.path.exists(nlp_model_name):
        network = joblib.load(nlp_model_name)
    else:
        network = MLPRegressor(solver='adam', hidden_layer_sizes=hidden_layer_sizes, max_iter=15000, tol=10e-6,
                               activation=activation)
        network.fit(vectors_train, label_train)
        joblib.dump(network, nlp_model_name)

    rt_pred_train = network.predict(vectors_train)
    rt_pred_test = network.predict(vectors_test)

    pred_train_origin, pred_test_origin = origin_log(rt_pred_train, rt_pred_test, label_max)

    mse_train = mean_squared_error(label_train, rt_pred_train)
    print("Mean Squared Error tarin:", mse_train)

    mse_train = mean_squared_error(retweets_train, pred_train_origin)
    print("Mean Squared Error tarin:", mse_train)

    mse_test = mean_squared_error(label_test, rt_pred_test)
    print("Mean Squared Error test:", mse_test)

    mse_test = mean_squared_error(retweets_test, pred_test_origin)
    print("Mean Squared Error test:", mse_test)

    # histogram orginalne i przewidziane log test
    plt.hist((label_test, rt_pred_test,))
    plt.figure()
    # histogram orginalne i przewidziane log train
    plt.hist((label_train, rt_pred_train,))
    plt.figure()
    # histogram orginalne i abs(orginalne - przewidziane) test
    plt.scatter(retweets_test, np.abs(retweets_test - pred_test_origin))
    plt.figure()
    # histogram orginalne i abs(orginalne - przewidziane) train
    plt.scatter(retweets_train, np.abs(retweets_train - pred_train_origin))
    plt.figure()
    # histogram orginalne i przewidziane rt test
    plt.hist((retweets_test, pred_test_origin,))
    plt.figure()
    # histogram orginalne i przewidziane rt train
    plt.hist((retweets_train, pred_train_origin,))

    plt.show()
