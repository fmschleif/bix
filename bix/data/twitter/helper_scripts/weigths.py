from keras import Model
from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix

from bix.data.twitter.base.utils import load_training_sentiment_data
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':

    #tok, y, padded_x, _, max_tweet_word_count, vocab_size = load_training_sentiment_data()

    model:Model = load_model('models/sentiment_conv_ep100.h5')
    weights = model.get_weights()
    print()
