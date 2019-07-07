from datetime import timedelta, date


class FetchConfig(object):
    def __init__(self) -> None:
        super().__init__()
        self.lang: str = 'en'

        """to_date >= from_date"""
        self.from_date: date = date.today() - timedelta(days=1)
        self.to_date: date = date.today()

        self.max_tweets_per_fetch = 0
