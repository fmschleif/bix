from random import shuffle

from keras_preprocessing import sequence
from keras_preprocessing.sequence import skipgrams

from bix.data.twitter.base.utils import load_training_sentiment_data, save_pickle

if __name__ == '__main__':
    tokenizer, y, padded_x, unpadded_x, max_tweet_word_count, vocab_size = load_training_sentiment_data()

    print('start shuffle')
    window_size=5

    #shuffle(unpadded_x)

    #flat_list = [item for sublist in unpadded_x for item in sublist]
    flat_list = []
    for sublist in unpadded_x:
        flat_list.extend(sublist)
        flat_list.extend([0]*window_size)

    print(f'start generating skip-grams | len:{len(flat_list)}')

    #ITERATIONS = 30000

    #grams_x = []
    #grams_y = []
    #for i, doc in enumerate(unpadded_x):
    sampling_table = sequence.make_sampling_table(vocab_size)
    data, labels = skipgrams(sequence=flat_list, vocabulary_size=vocab_size, window_size=window_size,
                             negative_samples=1., sampling_table=sampling_table)
    #grams_x.extend(data)
    #grams_y.extend(labels)

    #    if i % 1000 == 0:
    #        print(f'progress: {i / ITERATIONS*100}%')

    #    if i == ITERATIONS:
    #        break

    print(f'generated {len(data)} samples')
    save_pickle(data, 'tokenized/learn/grams_x.pickle')
    save_pickle(labels, 'tokenized/learn/grams_y.pickle')
