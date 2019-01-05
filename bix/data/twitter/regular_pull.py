import datetime
from collections import Counter
from time import sleep

from bix.data.twitter.twitter_retriever import TwitterRetriever

if __name__ == '__main__':
    time_to_wait = datetime.timedelta(days=1).total_seconds() + 1

    # tweets to request for each hashtag
    # after removing duplicates there are way less tweets
    tweets_to_pull = 10000
    while True:
        tr = TwitterRetriever()
        res = tr.search_hashtags(['#akk', '#merz'], count=tweets_to_pull, output_file='twitter.csv', lang='de')
        c = Counter()
        for e in map(lambda x: x[0], res):
            c[e] += 1
        print(f'pulled tweets: {c}')
        sleep(time_to_wait)
