import nltk
import pandas as pd

def load_from_csv(path, text_col):
    df = pd.read_csv("amazon1.csv")
    df = df.dropna(subset=[text_col])
    dflist = df[text_col].astype(str).tolist()
    text = " ".join(dflist)
    return text, dflist

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    words = [token for token in tokens if token not in stop_words]
    return words

def pos_tag(tokens):
    return nltk.pos_tag(tokens)

from collections import Counter
import matplotlib.pyplot as plt

def plot_pos_tags(pos_tags):
    tag_count = Counter(tag for word, tag in pos_tags)
    tags = list(tag_count.keys())
    count = list(tag_count.values())
    plt.figure(figsize=[9, 13])
    plt.bar(tags, count, color="Blue")
    plt.xlabel("tags")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

from wordcloud import WordCloud

def wordcloud(tokens):
    text = " ".join(tokens)
    wordcloud = WordCloud(height=500, width=500, background_color='White').generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.figure()

from textblob import TextBlob

def sentiment(tokens):
    text = " ".join(tokens)
    blob = TextBlob(text)
    sentiment = blob.sentiment
    print(f'{sentiment.polarity}')
    print(f'{sentiment.subjectivity}')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity(textlist):
    vectorizer = TfidfVectorizer()
    tfidf_mat = vectorizer.fit_transform(textlist)
    cos_sim = cosine_similarity(tfidf_mat, tfidf_mat)
    df_sim = pd.DataFrame(cos_sim, index=[f'Doc {i}' for i in range(len(textlist))], columns=[f'Doc {i}' for i in range(len(textlist))])
    return df_sim

path = "amazon1.csv"
text_col = "text"
text, dflist = load_from_csv(path, text_col)

df_sim = similarity(dflist)
print(df_sim)

tokens = preprocess(text)
print(f'{tokens[:10]}')

pos_tags = pos_tag(tokens)
print(f'{pos_tags[:10]}')

plot_pos_tags(pos_tags)

wordcloud(tokens)

sentiment(tokens)
