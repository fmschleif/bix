import pickle
from typing import List

from keras_preprocessing.text import Tokenizer


def tokenize(data: List[str], verbose: bool = False) -> Tokenizer:

    # 545409 is big
    t = Tokenizer(filters='!"„“…»«#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', num_words=25000)

    t.fit_on_texts(data)

    if verbose:
        print(f'indexsize: {len(t.word_counts)}')
        #print('wordcounts')
        #print(t.word_counts)
        #print('document_count')
        #print(t.document_count)
        #print('wordindex')
        #print(t.word_index)
        #print('word_docs')
        #print(t.word_docs)

        print('--------- ' + data[0][0] + ' ---------')
        for e, n in sorted(t.word_counts.items(), key=lambda x: x[1])[-9:]:
            print('\t' + str(e) + ': ' + str(n))

    return t


def save_tokenizer(tok: Tokenizer):
    from bix.twitter.base.utils import create_path_if_not_exists
    # saving
    with open(f'tok.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer() -> Tokenizer:
    # loading
    with open(f'tok.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer
