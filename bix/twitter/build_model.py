from datetime import date, timedelta

import numpy

from bix.twitter.analysis.analyse_sentiment_conv import train_model_convolutional
from bix.twitter.fetch.download_tweets import download_tweets_twint
from bix.twitter.fetch.fetch_config import FetchConfig
from bix.twitter.learn.learn_embeddings import learn_embeddings
from bix.twitter.preprocessing.preprocess import preprocess, tokenize_cleaned_tweets

if __name__ == '__main__':

    ## step 1: get data (using the sentiment data)
    #x, y = load_trainings_data() # x = tweets, y = sentiment

    # step 1: fetch tweets (for categorizing hashtags)
    hashtags = ['brexit', 'lol']
    config = FetchConfig()
    config.from_date = date.today() # fetch all tweets from today
    config.to_date = date.today() + timedelta(days=2) # to_date is exclusive
    config.max_tweets_per_fetch = 10
    tweets = download_tweets_twint(hashtags, config)

    # step 2: cleanup
    #cleaned_tweets = preprocess({'trainings_data': x}) # for sentiment data
    cleaned_tweets = preprocess(tweets)

    # step 3: tokenization
    tokenized_tweets = tokenize_cleaned_tweets(cleaned_tweets, create_tokenizer=True)
    encoded_categories = {hashtags[i]:i for i in range(len(hashtags))}
    y = []
    for hashtag, tweets in tokenized_tweets.items():
        for item in tweets:
            y.append(encoded_categories[hashtag])
    x = numpy.concatenate(list(tokenized_tweets.values()))

    # step 4: creating embeddings
    glove, word, sg = learn_embeddings(x, y, cleaned_tweets)

    # step 5: learn model
    train_model_convolutional(x,y, [word, glove, sg])