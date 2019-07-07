from bix.data.twitter.base.utils import load_csv, load_pickle
from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer
import numpy as np


def load_training_sentiment_data():
    t = load_tokenizer('learn')
    y = load_csv('tokenized/learn/lables.csv')
    padded_x = np.load('tokenized/learn/padded_x.npy')
    unpadded_x = load_pickle('tokenized/learn/unpadded_x.pickle')
    max_tweet_word_count = load_pickle('tokenized/learn/max_tweet_word_count.pickle')
    return t, y, padded_x, unpadded_x, max_tweet_word_count
