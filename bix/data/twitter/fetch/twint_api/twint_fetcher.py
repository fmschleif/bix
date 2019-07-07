from datetime import date, timedelta

from bix.data.twitter.fetch.abstract_fetcher import AbstractFetcher
from bix.data.twitter.fetch.fetch_config import FetchConfig

import twint

class TwintFetcher(AbstractFetcher):
    def _fetch_impl(self, query_string: str, date_: date, config: FetchConfig = FetchConfig()):

        c = twint.Config()

        c.Search = '#' + query_string
        c.Lang = config.lang
        c.Store_object = True
        c.Since = "{:%Y-%m-%d}".format(date_)
        c.Until = "{:%Y-%m-%d}".format(date_ + timedelta(days=1))

        twint.output.tweets_object = []
        twint.run.Search(c)
        result = twint.output.tweets_object
        tweets = [t.tweet for t in result]
        self.save_tweets(tweets, query_string, date_)
