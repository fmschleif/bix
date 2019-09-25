from datetime import date, timedelta

from bix.twitter.fetch.fetch_config import FetchConfig
from bix.twitter.fetch.twint_api.twint_fetcher import TwintFetcher

if __name__ == '__main__':
    hashtags = ['love', 'sad']
    # love, sad

    config = FetchConfig()
    config.to_date = date.today()
    config.from_date = date.today() - timedelta(days=1)

    tf = TwintFetcher()
    tf.fetch_many(hashtags, config)
