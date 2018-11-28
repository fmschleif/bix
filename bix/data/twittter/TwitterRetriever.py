import os
from typing import List

import twitter

from TwitterQueryResult import TwitterQueryResult
from datetime import *

class TwitterRetriever:
    def __init__(self, get_keys_from_env: bool = True) -> None:
        super().__init__()
        self.ACCESS_TOKEN = ''
        self.ACCESS_SECRET = ''
        self.CONSUMER_KEY = ''
        self.CONSUMER_SECRET = ''
        self.api: twitter.Api = None

        if get_keys_from_env:
            self.ACCESS_TOKEN = self.__get_access_token_from_env()
            self.ACCESS_SECRET = self.__get_access_secret_from_env()
            self.CONSUMER_KEY = self.__get_consumer_token_from_env()
            self.CONSUMER_SECRET = self.__get_consumer_secret_from_env()
            self.api = self.__init_api()

    @classmethod
    def create_with_keys(cls, access_token: str, access_secret: str, consumer_key: str, consumer_secret: str):
        tr = TwitterRetriever(searchForKeys=False)
        tr.ACCESS_TOKEN = access_token
        tr.ACCESS_SECRET = access_secret
        tr.CONSUMER_KEY = consumer_key
        tr.CONSUMER_SECRET = consumer_secret
        tr.api = tr.__init_api()

    def search_text(self, query_strings: List[str], start_date: date = None, end_date: date = None,
                    count: int = 15, lang: str = 'de') -> TwitterQueryResult:
        """
        Returns a TwitterQueryResult object with a to_map and to_csv method
        """
        self.__validate_date(start_date)
        self.__validate_date(end_date)

        ret = TwitterQueryResult()
        for s in query_strings:
            res = self.api.GetSearch(term=s, until=end_date, since=start_date, count=count, lang=lang)
            ret.results[s] = [t.text.split() for t in res]

        return ret

    def __get_access_token_from_env(self) -> str:
        return self.__get_env('TWITTER_ACCESS_TOKEN')

    def __get_access_secret_from_env(self) -> str:
        return self.__get_env('TWITTER_ACCESS_SECRET')

    def __get_consumer_token_from_env(self) -> str:
        return self.__get_env('TWITTER_CONSUMER_TOKEN')

    def __get_consumer_secret_from_env(self) -> str:
        return self.__get_env('TWITTER_CONSUMER_SECRET')

    @staticmethod
    def __get_env(var: str) -> str:
        env = os.getenv(var)
        if env:
            return env
        else:
            raise Exception('Environment variable "%s" not found! (hint: use the "create_with_keys" function instead)'
                            % var)

    def __init_api(self) -> twitter.Api:
        return twitter.Api(consumer_key=self.CONSUMER_KEY,
                           consumer_secret=self.CONSUMER_SECRET,
                           access_token_key=self.ACCESS_TOKEN,
                           access_token_secret=self.ACCESS_SECRET)

    @staticmethod
    def __validate_date(var:date):
        if var is None: return
        if var < date.today() - timedelta(days=7): raise Exception('date "%s" lies to far in the past (7 days)')

