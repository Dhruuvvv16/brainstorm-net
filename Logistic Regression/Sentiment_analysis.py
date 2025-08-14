from LogisticRegression import LogisticRegression1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
stemmer = PorterStemmer()

df = pd.read_csv('twitter_training.csv')
df.columns = ['er','sr', 'label', 'class']
df = df.drop(['er','sr'],axis=1)

df_positive = df[df['label'] == 'Positive']
df_negative = df[df['label'] == 'Negative'].sample(n=len(df_positive), random_state=42)
df_balanced = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_balanced['label'].value_counts())

X = df_balanced['class'].values
y = df_balanced['label'].values
y = np.array([1 if label == 'Positive' else 0 for label in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def process_tweet(tweet):
    tweet = str(tweet)
    tweet = re.sub(r'@\w+','',tweet)
    tweet = re.sub(r'\d','',tweet)
    tweet = re.sub(r'(https?://\S+|www\.\S+|\b\w+\.\w+/\S+)', '',tweet)
    tweet = re.sub(r'\.', '', tweet)
    tweet = re.sub(r'\b(?![ai]\b)\w\b','',tweet)
    tweet = re.sub(r"[^\w\s]", " ", tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords.words('english') and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean
def frequency(tweets,label):
    freq = {}
    for y, tweet in zip(label, tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            freq[pair] = freq.get(pair,0) + 1
    return freq
def extract_features(tweet, freqs, process_tweet=process_tweet):
    word_p = process_tweet(tweet)
    x = np.zeros(3)
    x[0] = 1
    for word in word_p:
        x[1] =+ freqs.get((word,1),0)
        x[2] =+ freqs.get((word,0),0)
    return x
