import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import pandas

from bix.twitter.base.utils import remove_duplicates, daterange, generate_csv_name, generate_hashtag_path, \
    save_csv, load_csv, create_path_if_not_exists
from bix.twitter.fetch.fetch_config import FetchConfig


class AbstractFetcher(object):

    def fetch_many(self, query_strings: List[str], config: FetchConfig = FetchConfig()) -> Dict[str, List[str]]:
        ret = {}
        for q in query_strings:
            ret[q] = self.fetch(q, config)
        return ret

    def fetch(self, query_string: str, config: FetchConfig = FetchConfig()) -> List[str]:
        ret = []
        for date_ in reversed(list(daterange(config.from_date, config.to_date))):
            ret = ret + self._fetch_impl(query_string, date_, config)
        return ret

    def _fetch_impl(self, query_string: str, date_: date, config: FetchConfig = FetchConfig()) -> List[str]:
        raise NotImplementedError()

    def get_max_tweets_per_session(self) -> Optional[int]:
        raise NotImplementedError()

    def save_tweets(self, tweets: List[str], query_string: str, date_: date):
        unique = remove_duplicates(tweets)
        dirname = generate_hashtag_path(query_string)
        filename = generate_csv_name(date_)
        fullfile = f'{dirname}_{filename}'

        #create_path_if_not_exists(dirname)

        if Path(fullfile).is_file():
            unique = load_csv(fullfile) + unique

            # remove duplicates again, to merge the results with the file
            unique = remove_duplicates(unique)

        save_csv(fullfile, unique)
        print(f'saved {len(unique)} unique tweets in {fullfile}')


