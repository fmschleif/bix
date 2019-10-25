from datetime import date, timedelta

from bix.twitter.fetch.abstract_fetcher import AbstractFetcher
from bix.twitter.fetch.fetch_config import FetchConfig

import twint

class TwintFetcher(AbstractFetcher):
    def _fetch_impl(self, query_string: str, date_: date, config: FetchConfig = FetchConfig()):

        c = twint.Config()

        c.Search = '#' + query_string
        c.Lang = config.lang
        c.Store_object = True
        c.Since = "{:%Y-%m-%d}".format(date_)
        c.Until = "{:%Y-%m-%d}".format(date_ + timedelta(days=1))
        c.Limit = config.max_tweets_per_fetch if config.max_tweets_per_fetch != 0 else None

        twint.run.Search(c)
        result = twint.output.tweets_list
        tweets = [t.tweet for t in result]
        self.save_tweets(tweets, query_string, date_)
        return tweets
