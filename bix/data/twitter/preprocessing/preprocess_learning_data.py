from typing import List

import pandas

from bix.data.twitter.base.utils import save_csv
from bix.data.twitter.preprocessing import cleanup

if __name__ == '__main__':
    path = 'learn/training.1600000.processed.noemoticon.csv'

    file_df = pandas.read_csv(path, header=None, encoding='ISO-8859-1')
    data = file_df.values.tolist()
    lables = [r[0] for r in data]
    tweets = [r[5] for r in data]

    options = {
        0: -1,
        2: 0,
        4: 1,
    }
    lables_int = list(map(lambda x: options[x], lables))


    # stemming and some other cleanup
    texts: List[str] = cleanup.clean_text(tweets, 'english')

    for i, v in enumerate(texts):
        if v == '' or v == ' ':
            print(f'deleted: \"{v}\"')
            del texts[i]
            del lables_int[i]

    save_csv('learn/lables.csv', lables_int)
    save_csv('learn/tweets.csv', texts)


