"""
@author: Jonas Burger <post@jonas-burger.de>
"""

from typing import List
import csv

from bix.data.twitter.config import CSV_DEFAULT_OUTPUT_DIR


class TwitterQueryResult:
    """
    Contains the results of a twitter-search.
    The Results can be extracted as a list by using the to_list function or written to a csv file by using the to_csv
    function and optionally specifying a filename parameter

    """
    def __init__(self) -> None:
        super().__init__()
        self.results: List[List[str]] = []

    def to_list(self) -> List[List[str]]:
        """
        :return: The results as a List with a List of tweets where the first element is the search-query and the
                 remaining elements contain the tweet
        """
        return self.results

    def to_csv(self, filename: str=CSV_DEFAULT_OUTPUT_DIR + 'twitter_data.csv'):
        """
        Write results to csv File by using ',' as seperator and '|' as quotechar
        example:
            #csv, csv, is, a, awesome, format
            #csv, this, is, a, example
        """
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for tweet_data in self.results:
                writer.writerow(tweet_data)
