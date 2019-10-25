from datetime import timedelta, date

import numpy
from keras.engine.saving import load_model

from bix.twitter.analysis.analyse_sentiment_conv import predict_model_convolutional
from bix.twitter.base.utils import load_model_mat
from bix.twitter.fetch.download_tweets import download_tweets_twint
from bix.twitter.fetch.fetch_config import FetchConfig

from bix.twitter.preprocessing.preprocess import preprocess, tokenize_cleaned_tweets

# before you run this script, set the execution folder for this script to where the tokenizer and the model are saved
if __name__ == '__main__':

    # step 1: fetch tweets (for categorizing hashtags)
    hashtags = ['brexit', 'lol'] # these should match the hashtags, the model was created with
    config = FetchConfig()
    config.from_date = date.today() # fetch all tweets from today
    config.to_date = date.today() + timedelta(days=1) # to_date is exclusive
    config.max_tweets_per_fetch = 10
    tweets = download_tweets_twint(hashtags, config)

    # step 2: cleanup
    cleaned_tweets = preprocess(tweets)

    # step 3: tokenization (using the tokenizer created in the build_model.py script)
    tokenized_tweets = tokenize_cleaned_tweets(cleaned_tweets)
    encoded_categories = {hashtags[i]: i for i in range(len(hashtags))}
    y = [] # eg. 0,0,0,1,1,1,1
    for hashtag, tweets in tokenized_tweets.items():
        for item in tweets:
            y.append(encoded_categories[hashtag])
    x = numpy.concatenate(list(tokenized_tweets.values()))


    # step 5: learn model
    model = load_model('sentiment_conv_ep100.h5')
    predictions = predict_model_convolutional(x, model=model, evaluate=True, y=y)
    print(predictions[0])









