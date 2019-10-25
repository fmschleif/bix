from typing import List, Dict

from bix.twitter.fetch.fetch_config import FetchConfig
from bix.twitter.fetch.twint_api.twint_fetcher import TwintFetcher


def download_tweets_twint(hashtags: List[str], config: FetchConfig) -> Dict[str, List[str]]:
    tf = TwintFetcher()
    return tf.fetch_many(hashtags, config)

