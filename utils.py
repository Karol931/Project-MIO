import numpy as np
import re

# Normalize quantile
def normalize_quantile(data):
    data = data.rank(pct=True)*100
    
    return data

# Normalize logaritmicly
def normalize_log(retweets):
    label_log = np.log(retweets + 1)
    label_max = max(label_log)
    label_log /= label_max
    
    return label_log, label_max

# Denormalize logarytmicly
def origin_log(rt_pred_train, rt_pred_test, label_max):
    pred_train_origin = np.exp(rt_pred_train * label_max) - 1
    pred_test_origin = np.exp(rt_pred_test * label_max) - 1

    return pred_train_origin, pred_test_origin

# Denormalize quantile
def origin_quant(rt_pred_train, rt_pred_test, data):
    rt_pred_test = np.clip(rt_pred_test, 0, 100)
    rt_pred_train = np.clip(rt_pred_train, 0, 100)

    pred_train_origin = np.quantile(data,rt_pred_train/100)
    pred_test_origin = np.quantile(data,rt_pred_test/100)

    return pred_train_origin, pred_test_origin


# Removes links from tweet
def remove_links(tweets_content):
    tweets_content = tweets_content.replace(to_replace="https?://\S+", value=" ", regex=True)
    return tweets_content

# Splits sufixow
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

# Removes tags
def remove_tags(tweets_content):
    tweets_content = tweets_content.replace(to_replace="(@)\s?(\w+)", value=" ", regex=True)

    return tweets_content

# Removes hashtags
def remove_hashtags(tweets_content):
    tweets_content = tweets_content.replace(to_replace="(#)\s?(\w+)", value=" ", regex=True)

    return tweets_content

# Removes many whitespaces
def remove_many_whitespaces(tweets_content):
    tweets_content = tweets_content.replace(to_replace="\s{2,}", value=" ", regex=True)

    return tweets_content

# Removes links to pictures
def remove_pictures(tweets_content):
    tweets_content = tweets_content.replace(to_replace="pic\.twitter\.com/\S+", value=" ", regex=True)

    return tweets_content

# Cleans tweets content
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

# Deletes tweets with 0 retweets and with less than 4 words
def delete_empty_tweets(df):
    for i in df.index:
        if(df["retweets"][i] == 0):
            df = df.drop(index=i)
        elif(len(str(df["content"][i]).split()) < 4):
            df = df.drop(index=i)
    return df

# Changes sentences to vectors
def get_tweets_vector_list(tweets_content, model):
    tweets_vectors = np.array([model.get_sentence_vector(tweet) for tweet in tweets_content])

    return tweets_vectors

