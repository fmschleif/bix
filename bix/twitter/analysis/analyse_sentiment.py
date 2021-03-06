import sys

import tensorflow
from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Dense, concatenate
from sklearn.model_selection import train_test_split

from bix.twitter.base.utils import load_model_mat, load_training_sentiment_data_small, load_pickle, load_csv

if __name__ == '__main__':
    args = sys.argv
    #    print(device_lib.list_local_devices())
    #    exit(0)

    tok, y, padded_x, _, max_tweet_word_count, vocab_size = load_training_sentiment_data_small()

    if 'test_all_data' in args:
        padded_x = load_pickle('tokenized/learn/padded_x.pickle')
        y = load_csv('learn/lables.csv')

    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.60)

    model_mat_word = load_model_mat('embedding_word')
    print(f'model_mat_word.shape: {model_mat_word[0].shape}')
    model_mat_glove = load_model_mat('embedding_glove')
    print(f'model_mat_glove.shape: {model_mat_glove[0].shape}')
    model_mat_skip_gram = load_model_mat('embedding_skip_gram')
    print(f'model_mat_skip_gram.shape: {model_mat_skip_gram[0].shape}')
    # model_mat_skip_gram = load_model_mat('embedding_skip_gram')

    # vocab_size = len(tok.word_index) + 1

    # model = Sequential()
    # model.add(Embedding(vocab_size, model_mat_word[0].shape[1], input_length=max_tweet_word_count,
    #                    weights=model_mat_word))
    # model.add(Flatten())

    # model.add(Dense(1, activation='sigmoid'))

    # word
    input1 = Input(shape=(max_tweet_word_count,))
    x1 = Embedding(vocab_size, model_mat_word[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat_word, trainable=False)(input1)
    x1 = Flatten()(x1)
    x1 = Model(inputs=input1, outputs=x1)

    # glove
    input2 = Input(shape=(max_tweet_word_count,))
    x2 = Embedding(vocab_size, model_mat_glove[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat_glove, trainable=False)(input2)
    x2 = Flatten()(x2)
    x2 = Model(inputs=input2, outputs=x2)

    # skip_gram
    input3 = Input(shape=(max_tweet_word_count,))
    x3 = Embedding(vocab_size, model_mat_skip_gram[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat_skip_gram, trainable=False)(input2)
    x3 = Flatten()(x3)
    x3 = Model(inputs=input2, outputs=x3)

    combined = concatenate([x1.output, x2.output, x3.output])

    z = Dense(10, activation="relu")(combined)
    z = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=[x1.input, x2.input, x3.input], outputs=z)

    # run_opts = tensorflow.RunOptions(report_tensor_allocations_upon_oom=True)
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # , options=run_opts)
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit([x_train, x_train, x_train], y_train, epochs=10, verbose=2)
    # evaluate the model
    loss, accuracy = model.evaluate([x_test, x_test], y_test, verbose=2)
    print('Accuracy: %f' % (accuracy * 100))
    # small: Accuracy: 99.763889
    # all data - Accuracy: 76.782812
    #3 embedding Layers:
    #

    print('finished')
