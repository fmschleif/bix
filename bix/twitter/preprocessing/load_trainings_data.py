from typing import List, Tuple
import pandas


def load_trainings_data(path: str = 'training.1600000.processed.noemoticon.csv') -> Tuple[List[str], List[int]]:

    file_df = pandas.read_csv(path, header=None, encoding='ISO-8859-1')
    data = file_df.values.tolist()
    lables = [r[0] for r in data]
    tweets = [r[5] for r in data]

    options = {
        0: 0,
        2: 0.5,
        4: 1,
    }
    lables_int = list(map(lambda x: options[x], lables))

    return tweets, lables_int


