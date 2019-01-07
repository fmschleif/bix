import os
from datetime import date, timedelta
import unittest

from bix.data.twitter.twitter_retriever import TwitterRetriever


class TestTwitterQueryResults(unittest.TestCase):
    """
    All test assume that the environment variables are set up accordingly or the config.py has been set up accordingly
        or they will fail

    The ResourceWarning returned by the unittests is because of the Resource-Model of the underlying Twitter API that
        is used
    """

    def test_retrieval_as_csv(self):
        tr = TwitterRetriever()
        tr.search_text(['Würzburg', 'Berlin'], count=10, lang='de', output_file='test.csv')
        os.remove('test.csv')

    def test_retrieval_as_list(self):
        tr = TwitterRetriever()
        result = tr.search_text(['Würzburg', 'Berlin'], count=10, lang='de')
        #self.assertEqual(len(result), 20) # most of the time 20, but it is not guaranteed

    def test_hashtag_search(self):
        tr = TwitterRetriever()
        result = tr.search_hashtags(['Würzburg', '#Berlin'], count=10, lang='de')
        for r in result:
            self.assertTrue(r[0].startswith('#'))

    def test_date(self):
        tr = TwitterRetriever()
        tr.search_text(['Würzburg'], start_date=date.today() - timedelta(days=7), end_date=date.today())

    def test_date_in_past(self):
        tr = TwitterRetriever()
        with self.assertRaises(Exception):
            tr.search_text(['Würzburg'], start_date=date(year=1970, month=1, day=1))
        with self.assertRaises(Exception):
            tr.search_text(['Würzburg'], end_date=date(year=1970, month=1, day=1))


if __name__ == '__main__':
    unittest.main()
