from typing import List
import csv

class TwitterQueryResult:
    def __init__(self) -> None:
        super().__init__()
        self.results = {}

    def to_map(self):
        return self.results

    def to_csv(self, filename='twitter_data.csv'):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k, tweets in self.results.items():
                for tweet in tweets:
                    writer.writerow([k] + tweet)


