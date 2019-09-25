from datetime import date

import twitter

from typing import Optional

from twitter import TwitterError

from bix.twitter.fetch.abstract_fetcher import AbstractFetcher
from bix.twitter.fetch.fetch_config import FetchConfig
from bix.twitter.fetch.twitter_api import TWITTER_CONFIG


class TwitterApiFetcher(AbstractFetcher):

    def __init__(self) -> None:
        super().__init__()
        self.ACCESS_TOKEN = TWITTER_CONFIG['TWITTER_ACCESS_TOKEN']
        self.ACCESS_SECRET = TWITTER_CONFIG['TWITTER_ACCESS_SECRET']
        self.CONSUMER_KEY = TWITTER_CONFIG['TWITTER_CONSUMER_TOKEN']
        self.CONSUMER_SECRET = TWITTER_CONFIG['TWITTER_CONSUMER_SECRET']
        self.api: twitter.Api = twitter.Api(consumer_key=self.CONSUMER_KEY,
                                            consumer_secret=self.CONSUMER_SECRET,
                                            access_token_key=self.ACCESS_TOKEN,
                                            access_token_secret=self.ACCESS_SECRET)

    def _fetch_impl(self, query_string: str, date_: date, config: FetchConfig = FetchConfig()):

        results = []
        max_tweets = config.max_tweets_per_fetch
        if max_tweets == 0:
            max_tweets = 20000
        tweet_amount = max_tweets

        last_max_id = None
        while tweet_amount > 0:
            statuses = None
            try:
                statuses = self.api.GetSearch(term='#' + query_string,
                                              until=date_,
                                              since=date_,
                                              count=tweet_amount,
                                              lang=config.lang,
                                              max_id=last_max_id,
                                              result_type='recent')
            except TwitterError as e:
                if e.message[0]['code'] == 88:  # Rate limit exceeded
                    print(f"Warning: aborting pulling tweets for query '{query_string}' after {int(max_tweets - tweet_amount)} "
                          f"tweets, because of error: '{e.message[0]['message']}'")
                    break
                else:
                    raise e
            if len(statuses) <= 1:
                print(f"Warning: aborting pulling tweets for query '{query_string}' after {int(max_tweets - tweet_amount)} tweets, "
                      f"because there are no more tweets available")
                break  # out of results
            results.extend([t.text for t in statuses])
            tweet_amount = tweet_amount - len(statuses)
            last_max_id = statuses[-1].id_str

        results.reverse()

        #results = self.remove_duplicates(results)
        for i,e in enumerate(results):  # remove 'RT' from tweets (retweets)
            if e.startswith('RT'):
                results[i] = e[2:]

        self.save_tweets(results, query_string, date_)

        return results

    def get_max_tweets_per_session(self) -> Optional[int]:
        return 20000
