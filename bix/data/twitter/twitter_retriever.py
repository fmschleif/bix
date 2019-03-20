"""
@author: Jonas Burger <post@jonas-burger.de>
"""
import os
from collections import defaultdict
from functools import reduce

import keras as keras
import pandas
import twitter

from pathlib import Path
from typing import List, Tuple, Dict
from datetime import *

from keras_preprocessing.text import Tokenizer
from numpy.core.multiarray import ndarray
from twitter import Status, TwitterError
from bix.data.twitter.config import TWITTER_CONFIG


class TwitterRetriever:
    """ TwitterRetriever
    The TwitterRetriever class manages Authentication and provides a use-case oriented interface to the twitter-API.
    For it to work it requires all twitter authentication keys (more information in the parameters section).

    Parameters
    ----------
    search_for_keys: bool (default = True)
        If this parameter is True the constructor searches for the twitter authentication keys in the config.py file or
        environment variables (TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, TWITTER_CONSUMER_TOKEN,
        TWITTER_CONSUMER_SECRET).
        If you want to supply the twitter authentication keys directly you should use the static create_with_keys
        function to create this class.

    Notes
    -----
    -Duplicates are automatically removed
    -If the output file already exists the results and the file is merged, but the function still just returns the
        current queries results
    -The twitter standard API only allows 180 requests every 15 minutes and 1 request can only contain up to 100 tweets.
        If you want to request as much tweets as possible 18000 is a good 'count'-parameter for 1 query
    -If you specify a start or a end date that is older than 7 days a Exception is raised
    -If you request more tweets than your twitter-contract allows, a warning is displayed and the search function
        continues with all the data that its got
    -The search functions always start with the newest tweets first
    -All search functions return a 2d List containing the tweets split by whitespace, starting with the hashtag eg.:
        [['#python', '#python', 'is', 'cool'], ['#python', 'I', 'like', '#python']]

    References
    --------
    More information about the twitter api: https://developer.twitter.com/en/docs/tweets/search/overview

    Examples
    --------
    >>> # imports
    >>> from bix.data.twitter.twitter_retriever import TwitterRetriever
    >>> # construction
    >>> tr = TwitterRetriever() # provided the keys are set in the config.py file
    >>> tr = TwitterRetriever.create_with_keys("access_token", "access_secret", "consumer_key", "consumer_secret")
    >>> # usage
    >>> result = tr.search_hashtags(['#python', '#FHWS'], count=100)
    >>> assert result[0][0] == '#python' or result[0][0] == '#FHWS'
    >>> for e in result:
    >>>     print(e)
    >>>
    >>> res = tr.search_text(['python is awesome'], count=100)
    >>> res = tr.search_hashtags(['#python'], count=100, output_file='twitter.csv', lang='de')
    >>> res = tr.search_hashtags(['#python'], count=100, start_date=date.today() - timedelta(days=7),
    >>>                          end_date=date.today())

    """

    def __init__(self, search_for_keys: bool = True) -> None:

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

        tr = TwitterRetriever(search_for_keys=False)
        tr.ACCESS_TOKEN = access_token
        tr.ACCESS_SECRET = access_secret
        tr.CONSUMER_KEY = consumer_key
        tr.CONSUMER_SECRET = consumer_secret
        tr.api = tr.init_api()
        return tr

    def search_text(self, query_strings: List[str], start_date: date = None,  # date.today() - timedelta(days=7)
                    end_date: date = None,  # date.today(),
                    count: int = 15, lang: str = 'de', output_file=None, newest_first: bool=False) -> List[List[str]]:
        self.validate_date(start_date)
        self.validate_date(end_date)

        results = []
        for i, s in enumerate(reversed(query_strings) if not newest_first else query_strings):

            tweet_amount = count
            last_max_id = None
            while tweet_amount > 0:
                statuses = None
                try:
                    statuses = self.api.GetSearch(term=s, until=end_date, since=start_date, count=tweet_amount,
                                                  lang=lang,
                                                  max_id=last_max_id, result_type='recent')
                except TwitterError as e:
                    if e.message[0]['code'] == 88:  # Rate limit exceeded
                        print(f"Warning: aborting pulling tweets for query '{s}' after {int(count - tweet_amount)} "
                              f"tweets, because of error: '{e.message[0]['message']}'")
                        break
                    else:
                        raise e
                if len(statuses) <= 1:
                    print(f"Warning: aborting pulling tweets for query '{s}' after {int(count - tweet_amount)} tweets, "
                          f"because there are no more tweets available")
                    break  # out of results
                results.extend([[s] + t.text.split() for t in statuses])
                tweet_amount = tweet_amount - len(statuses)
                last_max_id = statuses[-1].id_str

        if not newest_first:
            results.reverse()

        results = self.remove_duplicates(results)

        if output_file is not None:
            results2 = results
            if Path(output_file).is_file():
                file_df = pandas.read_csv(output_file, header=None)
                results2 = (file_df.values.tolist() + results2) if not newest_first else \
                    results + file_df.values.tolist()
                results2 = self.remove_duplicates(
                    results2)  # remove duplicates again, to merge the results with the file
            df = pandas.DataFrame(results2)
            df.to_csv(output_file, encoding='utf-8', header=False, index=False)

        return results

    def search_hashtags(self, hashtags: List[str], start_date: date = None, end_date: date = None,
                        count: int = 15, lang: str = 'de', output_file=None, newest_first: bool=False) \
            -> List[List[str]]:
        return self.search_text(query_strings=[h if h.startswith('#') else '#' + h for h in hashtags],
                                start_date=start_date, end_date=end_date, count=count, lang=lang,
                                output_file=output_file, newest_first=newest_first)

    def get_access_token(self) -> str:
        return self.get_config_entry('TWITTER_ACCESS_TOKEN') or TWITTER_CONFIG['TWITTER_ACCESS_TOKEN']

    def get_access_secret(self) -> str:
        return self.get_config_entry('TWITTER_ACCESS_SECRET') or TWITTER_CONFIG['TWITTER_ACCESS_SECRET']

    def get_consumer_token(self) -> str:
        return self.get_config_entry('TWITTER_CONSUMER_TOKEN') or TWITTER_CONFIG['TWITTER_CONSUMER_TOKEN']

    def get_consumer_secret(self) -> str:
        return self.get_config_entry('TWITTER_CONSUMER_SECRET') or TWITTER_CONFIG['TWITTER_CONSUMER_SECRET']

    def get_config_entry(self, var: str) -> str:
        result = os.getenv(var) or TWITTER_CONFIG[var]
        if result:
            return result
        else:
            raise Exception('Variable "%s" not found in config! (hint: configure config.py, set environment variables'
                            'or use the "create_with_keys" function instead)'
                            % var)

    def init_api(self) -> twitter.Api:
        return twitter.Api(consumer_key=self.CONSUMER_KEY,
                           consumer_secret=self.CONSUMER_SECRET,
                           access_token_key=self.ACCESS_TOKEN,
                           access_token_secret=self.ACCESS_SECRET)

    def remove_duplicates(self, lst: List[List[str]]):
        for i,e in enumerate(lst):  # remove 'RT' from tweets (retweets)
            if e[1] == 'RT':
                del e[1]
            lst[i] = [s for s in e if not pandas.isna(s)]
        unique = []
        [unique.append(item) for item in lst if item not in unique]
        return unique

    def validate_date(self, var: date):
        if var is None:
            return
        if var < date.today() - timedelta(days=7):
            raise Exception('date "%s" lies to far in the past (7 days)')

    @classmethod
    def split_result_list_by_label(cls, data: List[List[str]]) -> Dict[str, List[List[str]]]:
        # dont, only after
        label_dict = defaultdict(list)
        for e in data:
            label_dict[e[0]].append(e)
        return label_dict


    @classmethod
    def tokenize_and_vectorize(cls, data: List[List[str]]) -> (ndarray, List[str]):
        transformed_data:List[str] = [' '.join([ee for ee in e[1:] if not pandas.isna(ee)]) for e in data]

        # add spaces after sc
        #tfidf
        # add vectorize function
        # numpy array anzahl
        # split label vector
        #
        t = Tokenizer(filters='!"„“…»«#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        t.fit_on_texts(transformed_data)

        #print('wordcounts')
        #print(t.word_counts)
        #print('document_count')
        #print(t.document_count)
        #print('wordindex')
        #print(t.word_index)
        #print('word_docs')
        #print(t.word_docs)

        print('--------- ' + data[0][0] + ' ---------')
        for e, n in sorted(t.word_counts.items(), key=lambda x: x[1]):
            print('\t' + str(e) + ': ' + str(n))

        mat = t.texts_to_matrix(transformed_data, mode='tfidf')
        print('dims: ' + str(mat.shape))

        return mat, transformed_data
