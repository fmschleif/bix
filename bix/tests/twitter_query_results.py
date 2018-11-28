import os
from datetime import date, timedelta
from bix.data.twittter.TwitterRetriever import TwitterRetriever
import unittest


class TestTwitterQueryResults(unittest.TestCase):
    """
    All test assume that the environment variables are set up accordingly or they will fail

    The ResourceWarning returned by the unittests is because of the Resource-Model of the Twitter API i use
    """

    def test_retrival_as_csv(self):
        tr = TwitterRetriever()
        result = tr.search_text(['Würzburg', 'Berlin'], start_date=date.today() - timedelta(days=7),
                                end_date=date.today(), count=10, lang='de')
        result.to_csv('test.csv')
        os.remove('test.csv')

    def test_retrival_as_map(self):
        tr = TwitterRetriever()
        result = tr.search_text(['Würzburg', 'Berlin'], start_date=date.today() - timedelta(days=7),
                                end_date=date.today(), count=10, lang='de')
        result_map = result.to_map()
        self.assertTrue('Würzburg' in result_map)
        self.assertTrue('Berlin' in result_map)
        self.assertEqual(len(result_map['Würzburg']), 10)
        self.assertEqual(len(result_map['Berlin']), 10)

    def test_date_in_past(self):
        tr = TwitterRetriever()
        with self.assertRaises(Exception):
            tr.search_text(['Würzburg'], start_date=date(year=1970, month=1, day=1))
        with self.assertRaises(Exception):
            tr.search_text(['Würzburg'], end_date=date(year=1970, month=1, day=1))


if __name__ == '__main__':
    unittest.main()
