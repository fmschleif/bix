from typing import List

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from bix.data.twitter.base.utils import encode_embed_docs


class EmbeddingAbstract:
    def __init__(self, tokenizer: Tokenizer, x: List[str], y: List[int]) -> None:
        self.vocab_size = len(tokenizer.word_index) + 1
        self.tokenizer = tokenizer
        self.max_tweet_word_count = max([len(e.split()) for e in x])

        self.x, self.unpadded_x = self._prepare_docs(x)
        self.y = y

        self.model = None

    def create_embedding(self):
        print('preparing...')
        self.prepare()
        print('defining model...')
        self.define_model()
        print('learning...')
        self.learn()
        return self.get_weights()

    def prepare(self):
        raise NotImplementedError()

    def define_model(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def _prepare_docs(self, x: List[str]):
        return encode_embed_docs(x, self.max_tweet_word_count)
