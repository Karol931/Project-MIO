import fasttext
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from divide_data import load_tweet_data
from utils import get_tweets_vector_list, normalize_log, origin_log, origin_quant

def load_fasttext_model(df, fasttext_model_name: str, model_size: int):
    if os.path.exists(fasttext_model_name):
        model = fasttext.load_model(fasttext_model_name)
    else:
        df.to_csv("tweet_words.txt", columns=["content"], header=None, index=False)
        model = fasttext.train_unsupervised('./tweet_words.txt', dim=model_size, model='skipgram', lr=0.01, lrUpdateRate=100)
        model.save_model(fasttext_model_name)
    
    return model


def make_nlp_name(hidden_layer_sizes: tuple, activation: str, normalize: str, model_size: int):
    out = "model_" + str(model_size) + "_mlp_"
    for hls in hidden_layer_sizes:
        out += str(hls) + "_"
    out += activation + "_"
    out += normalize
    out += ".joblib"
    
    return out


def create_model(normalize :str):
    text_train, text_test, retweets_train, retweets_test, quant_rt_train, quant_rt_test, log_rt_train, log_rt_test  = load_tweet_data()

    retweets = np.concatenate((retweets_train, retweets_test))
    content = np.concatenate((text_train, text_test))

    df = pd.DataFrame(data={'content': content, 'retweets': retweets})

    model_size = 400  #zmienna do wytestowania
    fasttext_model_name = "saved_model_trumptweets_" + str(model_size) + ".bin"
    model = load_fasttext_model(df, fasttext_model_name, model_size)

    vectors_train = get_tweets_vector_list(text_train, model)
    vectors_test = get_tweets_vector_list(text_test, model)

    #zmienne do wytestowania
    hidden_layer_sizes = (200,200,200)
    activation = 'relu'

    nlp_model_name = make_nlp_name(hidden_layer_sizes, activation, normalize, model_size)
    print(nlp_model_name)

    
    _, label_max = normalize_log(retweets)

    if normalize == "log":
        label_train = log_rt_train
        label_test = log_rt_test
    else: # if "quant"
        label_train = quant_rt_train
        label_test = quant_rt_test

    if os.path.exists(nlp_model_name):
        network = joblib.load(nlp_model_name)
    else:
        network = MLPRegressor(solver='adam', hidden_layer_sizes=hidden_layer_sizes, max_iter=15000, tol=10e-8,
                               activation=activation)
        network.fit(vectors_train, label_train)
        joblib.dump(network, nlp_model_name)

    rt_pred_train = network.predict(vectors_train)
    rt_pred_test = network.predict(vectors_test)

    if normalize == "log":
        pred_train_origin, pred_test_origin = origin_log(rt_pred_train, rt_pred_test, label_max)
    else: # if "quant"
        pred_train_origin, pred_test_origin = origin_quant(rt_pred_train, rt_pred_test, retweets)

    mse_train = mean_squared_error(label_train, rt_pred_train)
    print("Mean Squared Error tarin:", mse_train)

    mse_train = mean_squared_error(retweets_train, pred_train_origin)
    print("Mean Squared Error tarin:", mse_train)

    mse_test = mean_squared_error(label_test, rt_pred_test)
    print("Mean Squared Error test:", mse_test)

    mse_test = mean_squared_error(retweets_test, pred_test_origin)
    print("Mean Squared Error test:", mse_test)

    # histogram orginalne i przewidziane log test
    plt.hist((label_test, rt_pred_test))
    plt.xlabel("Znormalizowane liczby retweetów")
    plt.ylabel("Liczba wyników")
    plt.figure()
    # histogram orginalne i przewidziane log train
    plt.hist((label_train, rt_pred_train,))
    plt.xlabel("Znormalizowane liczby retweetów")
    plt.ylabel("Liczba wyników")
    plt.figure()
    if normalize == "log":
        # histogram orginalne i abs(orginalne - przewidziane) test
        plt.scatter(retweets_test, abs(retweets_test - pred_test_origin))
        plt.xlabel("Liczba retweetów")
        plt.ylabel("Błąd przewidzaniej liczby retweetów")
        plt.figure()
        # histogram orginalne i abs(orginalne - przewidziane) train
        plt.scatter(retweets_train, abs(retweets_train - pred_train_origin))
        plt.xlabel("Liczba retweetów")
        plt.ylabel("Błąd przewidzaniej liczby retweetów")
        plt.figure()
    else: # if "quant"
        # histogram orginalne i abs(orginalne - przewidziane) quant test
        plt.scatter(quant_rt_test, abs(quant_rt_test - rt_pred_test))
        plt.xlabel("Liczba retweetów w skali kwantylowej")
        plt.ylabel("Błąd przewidzaniej liczby retweetów w skali kwantylowej")
        plt.figure()
        # histogram orginalne i abs(orginalne - przewidziane) quant train
        plt.scatter(quant_rt_train, abs(quant_rt_train - rt_pred_train))
        plt.xlabel("Liczba retweetów w skali kwantylowej")
        plt.ylabel("Błąd przewidzaniej liczby retweetów w skali kwantylowej")
        plt.figure()
    # histogram orginalne i przewidziane rt test
    plt.hist((retweets_test, pred_test_origin,))
    plt.figure()
    # histogram orginalne i przewidziane rt train
    plt.hist((retweets_train, pred_train_origin,))

    plt.show()