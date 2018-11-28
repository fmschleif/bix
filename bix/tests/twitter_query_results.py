import os
from datetime import date, timedelta
<<<<<<< HEAD:bix/tests/twitter_query_results.py
from bix.data.twittter.TwitterRetriever import TwitterRetriever
=======
>>>>>>> e754b1dd1510b7e8945fd433a3c5f1226bfc9848:bix/tests/test_twitter_query_results.py
import unittest

from bix.data.twitter.twitter_retriever import TwitterRetriever


class TestTwitterQueryResults(unittest.TestCase):
    """
    All test assume that the environment variables are set up accordingly or they will fail

    The ResourceWarning returned by the unittests is because of the Resource-Model of the Twitter API i use
    """

    def test_retrieval_as_csv(self):
        tr = TwitterRetriever()
        result = tr.search_text(['Würzburg', 'Berlin'], start_date=date.today() - timedelta(days=7),
                                end_date=date.today(), count=10, lang='de')
        result.to_csv('test.csv')
        os.remove('test.csv')

    def test_retrieval_as_list(self):
        tr = TwitterRetriever()
        result = tr.search_text(['Würzburg', 'Berlin'], start_date=date.today() - timedelta(days=7),
                                end_date=date.today(), count=10, lang='de')
        result_map = result.to_list()
        for i in range(0, 10):
            self.assertTrue('Würzburg', result_map[i][0])
        for i in range(10, 20):
            self.assertTrue('Berlin', result_map[i][0])

    def test_hashtag_search(self):
        tr = TwitterRetriever()
        result = tr.search_hashtags(['Würzburg', '#Berlin'], start_date=date.today() - timedelta(days=7),
                                end_date=date.today(), count=10, lang='de')
        result_list = result.to_list()
        for r in result_list:
            self.assertTrue(r[0].startswith('#'))

    def test_date_in_past(self):
        tr = TwitterRetriever()
        with self.assertRaises(Exception):
            tr.search_text(['Würzburg'], start_date=date(year=1970, month=1, day=1))
        with self.assertRaises(Exception):
            tr.search_text(['Würzburg'], end_date=date(year=1970, month=1, day=1))


if __name__ == '__main__':
    unittest.main()
