import string
import pandas as pd
import numpy as np
# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy
import re

nlp = spacy.load('en_core_web_sm')


def sentvec(s):
    sent = nlp(s)
    return np.mean([w.vector for w in sent], axis=0)

def cosine(v1, v2):
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0.0

df = pd.read_csv("trumptweets.csv")

print(np.mean(df['favorites']))
print(np.median(df['favorites']))

for text in df.loc[0:10,'content']:
    print(re.sub(r'http://\S+', '', text))
    # print(text.translate(str.maketrans('', '', string.punctuation)).lower())

v1 = sentvec("i like apples")
v2 = sentvec("i like oranges")
v3 = sentvec("fuck my ass")
print(cosine(v1,v2))
print(cosine(v1,v3))
print(cosine(v2,v3))



# print(doc[0])
# print(vec("My") == doc[0].vector)
# print(vec("never"))
# print(vec("ass"))


# Analyze syntax
# nouns = [chunk for chunk in doc.noun_chunks]
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
# print(nouns[0])
# print(nouns[0].vector)
# print(nlp.vocab.get_vector("apple"))

# Find named entities, phrases and concepts
# for token in doc:
#     print(token.has_vector)


#
# # Wyświetlanie danych w DataFramie
# data = df.loc[1,'content']
# # print(data)
# for d in data:
#     print(d)
#     print(type(d))
# print(type(data))
