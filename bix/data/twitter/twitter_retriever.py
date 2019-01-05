"""
@author: Jonas Burger <post@jonas-burger.de>
"""

import os
from pathlib import Path
from typing import List, Tuple

import pandas
import twitter
import numpy as np

from datetime import *

from twitter import Status, TwitterError

from bix.data.twitter.config import TWITTER_CONFIG


class TwitterRetriever:
    """
    The TwitterRetriever class manages Authentication and provides a use-case oriented interface to the twitter-API.
    """

    def __init__(self, search_for_keys: bool = True) -> None:
        """
        The constructor of the TwitterRetriever class. Simply constuct this class ( TwitterRetriever() ) if the
        authentication keys have been already set.
        :param search_for_keys:
            Search for Twitter Authentication keys in environment variables and config.py (default: True)
        :exception if search_for_keys is True and the keys can't be found a Exception is raised
        """
        super().__init__()
        self.ACCESS_TOKEN = ''
        self.ACCESS_SECRET = ''
        self.CONSUMER_KEY = ''
        self.CONSUMER_SECRET = ''
        self.api: twitter.Api = None

        if search_for_keys:
            self.ACCESS_TOKEN = self.get_access_token()
            self.ACCESS_SECRET = self.get_access_secret()
            self.CONSUMER_KEY = self.get_consumer_token()
            self.CONSUMER_SECRET = self.get_consumer_secret()
            self.api = self.init_api()

    @classmethod
    def create_with_keys(cls, access_token: str, access_secret: str, consumer_key: str, consumer_secret: str):
        """
        An alternative way of constructing this class without having set the Twitter Authentication keys beforehand.
        :param access_token: TWITTER_ACCESS_TOKEN
        :param access_secret: TWITTER_ACCESS_SECRET
        :param consumer_key: TWITTER_CONSUMER_KEY
        :param consumer_secret: TWITTER_CONSUMER_SECRET
        :return: a new TwitterRetriever instance
        """
        tr = TwitterRetriever(searchForKeys=False)
        tr.ACCESS_TOKEN = access_token
        tr.ACCESS_SECRET = access_secret
        tr.CONSUMER_KEY = consumer_key
        tr.CONSUMER_SECRET = consumer_secret
        tr.api = tr.init_api()

    def search_text(self, query_strings: List[str], start_date: date = None,# date.today() - timedelta(days=7)
                    end_date: date = None, #date.today(),
                    count: int = 15, lang: str = 'de', output_file=None) -> List[List[str]]:
        """
        Search for a number of strings on Twitter.

        :param query_strings: a list of strings to search for on twitter
        :param start_date: limit the search by excluding results before a date (default: 7 days ago)
        :param end_date: limit the search by excluding results past a date (default: today)
        :param count: number of results for each query_string (default 15)
        :param lang: iso country code (default: de)
        :return: Returns a TwitterQueryResult object with a to_list and to_csv method
        """
        self.validate_date(start_date)
        self.validate_date(end_date)

        results = []
        for i, s in enumerate(query_strings):

            tweet_amount = count
            last_max_id = None
            while tweet_amount > 0:
                statuses = None
                try:
                    statuses = self.api.GetSearch(term=s, until=end_date, since=start_date, count=tweet_amount, lang=lang,
                                             max_id=last_max_id, result_type='recent')
                except TwitterError as e:
                    if e.message[0]['code'] == 88: # Rate limit exceeded
                        print(f"Warning: aborting pulling tweets for query '{s}' after {count - tweet_amount} tweets, "
                              f"because of error: '{e.message[0]['message']}'")
                        break
                    else:
                        raise e
                if len(statuses) == 0:
                    print(f"Warning: aborting pulling tweets for query '{s}' after {count - tweet_amount} tweets, "
                          f"because there are no more available")
                    break  # out of results
                results.extend([[s] + t.text.split() for t in statuses])
                tweet_amount = tweet_amount - len(statuses)
                last_max_id = statuses[-1].id_str

        results = self.remove_duplicates(results)

        if output_file is not None:
            results2 = results
            if Path(output_file).is_file():
                file_df = pandas.read_csv(output_file, header=None)
                results2 = results2 + file_df.values.tolist()
                results2 = self.remove_duplicates(results2) # remove duplicates again, to merge the results with the file
            df = pandas.DataFrame(results2)
            df.to_csv(output_file, encoding='utf-8', header=False, index=False)

        return results

    def search_hashtags(self, hashtags: List[str], start_date: date = None, end_date: date = None,
                        count: int = 15, lang: str = 'de', output_file=None) -> List[List[str]]:
        """
        Search for a number of hashtags on Twitter.

        :param hashtags:
            a list of hashtags to search for on twitter. It doesn't mater if the hashtags start with # or not
        :param start_date: limit the search by excluding results before a date (default: 7 days ago)
        :param end_date: limit the search by excluding results past a date (default: today)
        :param count: number of results for each query_string (default 15)
        :param lang: iso country code (default: de)
        :return: Returns a TwitterQueryResult object with a to_list and to_csv method
        :exception raises a Exception if the start or end date is more than 7 days in the past or in the future
        """
        return self.search_text(query_strings=[h if h.startswith('#') else '#' + h for h in hashtags],
                                start_date=start_date, end_date=end_date, count=count, lang=lang,
                                output_file=output_file)

    def get_access_token(self) -> str:
        """
        helper function for searching Twitter authentication keys
        :return: key or None
        """
        return self.get_config_entry('TWITTER_ACCESS_TOKEN') or TWITTER_CONFIG['TWITTER_ACCESS_TOKEN']

    def get_access_secret(self) -> str:
        """
        helper function for searching Twitter authentication keys
        :return: key or None
        """
        return self.get_config_entry('TWITTER_ACCESS_SECRET') or TWITTER_CONFIG['TWITTER_ACCESS_SECRET']

    def get_consumer_token(self) -> str:
        """
        helper function for searching Twitter authentication keys
        :return: key or None
        """
        return self.get_config_entry('TWITTER_CONSUMER_TOKEN') or TWITTER_CONFIG['TWITTER_CONSUMER_TOKEN']

    def get_consumer_secret(self) -> str:
        """
        helper function for searching Twitter authentication keys
        :return: key or None
        """
        return self.get_config_entry('TWITTER_CONSUMER_SECRET') or TWITTER_CONFIG['TWITTER_CONSUMER_SECRET']

    def get_config_entry(self, var: str) -> str:
        """
        helper function for searching Twitter authentication keys
        :return: key or None
        :exception Exception
        """
        result = os.getenv(var) or TWITTER_CONFIG[var]
        if result:
            return result
        else:
            raise Exception('Variable "%s" not found in config! (hint: configure config.py, set environment variables'
                            'or use the "create_with_keys" function instead)'
                            % var)

    def init_api(self) -> twitter.Api:
        """
        helper function for constructing a twitter.Api instance
        :return: a new twitter.Api instance
        """
        return twitter.Api(consumer_key=self.CONSUMER_KEY,
                           consumer_secret=self.CONSUMER_SECRET,
                           access_token_key=self.ACCESS_TOKEN,
                           access_token_secret=self.ACCESS_SECRET)

    def remove_duplicates(self, lst: List[List[str]]):
        for e in lst: # remove 'RT' from tweets (retweets)
            if e[1] == 'RT':
                del e[1]
        s = set([tuple(e) for e in lst])
        return list([list(e) for e in s])

    def validate_date(self, var: date):
        """
        helper function for making sure a date is not more than 7 days in the past and not in the future
        :param var: date
        :exception Exception
        """
        if var is None:
            return
        if var < date.today() - timedelta(days=7):
            raise Exception('date "%s" lies to far in the past (7 days)')
