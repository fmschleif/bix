import datetime
from collections import Counter
from time import sleep
from bix.data.twitter.twitter_retriever import TwitterRetriever

""" regular pull script
A small script that tries to gather as much tweets as possible for the given hashtags every 24h and saves them in the
output file (merges results)
"""
hashtags = ['#akk', '#merz']
tweets_to_pull = 20000 / len(hashtags)
output_file = 'twitter.csv'

if __name__ == '__main__':
    time_to_wait = datetime.timedelta(days=1).total_seconds() + 1

    # tweets to request for each hashtag
    # after removing duplicates there are way less tweets
    while True:
        tr = TwitterRetriever()
        res = tr.search_hashtags(hashtags, count=tweets_to_pull, output_file=output_file, lang='de')
        c = Counter()
        for e in map(lambda x: x[0], res):
            c[e] += 1
        print(f'unique tweets: {c.items()}')
        sleep(time_to_wait)
