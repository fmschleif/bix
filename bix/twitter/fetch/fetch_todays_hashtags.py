from datetime import date, timedelta

from bix.twitter.fetch.fetch_config import FetchConfig
from bix.twitter.fetch.twint_api.twint_fetcher import TwintFetcher

if __name__ == '__main__':
    hashtags = ['brexit']

    config = FetchConfig()
    config.to_date = date.today() + timedelta(days=1)
    config.from_date = date.today()

    tf = TwintFetcher()
    tf.fetch_many(hashtags, config)
