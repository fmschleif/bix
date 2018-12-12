import datetime
from time import sleep

from bix.data.twitter.twitter_retriever import TwitterRetriever

if __name__ == '__main__':
    time_to_wait = datetime.timedelta(days=1).total_seconds() + 1
    tweets_to_pull = 10000 # as much as possible or 1000000
    max_tweets = None # to make sure that we don't pull the same tweets multiple times
    while True:
        tr = TwitterRetriever()
        _, max_tweets = tr.search_hashtags(['akk', 'merz'], count=int(tweets_to_pull / 2), output_file='twitter.csv',
                                           max_ids=max_tweets)
        sleep(time_to_wait)
