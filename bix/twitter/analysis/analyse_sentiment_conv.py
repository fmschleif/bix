import sys

import tensorflow
from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Dense, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout
from sklearn.model_selection import train_test_split

from bix.twitter.base.utils import load_model_mat, load_training_sentiment_data_small, load_pickle, load_csv, \
    load_training_sentiment_data

if __name__ == '__main__':
    args = sys.argv
    #    print(device_lib.list_local_devices())
    #    exit(0)

    tok, y, padded_x, _, max_tweet_word_count, vocab_size = load_training_sentiment_data()

    #if 'test_all_data' in args:
    #    padded_x = load_pickle('tokenized/learn/padded_x.pickle')
    #    y = load_csv('learn/lables.csv')

    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.2, random_state=5)

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
    input = Input(shape=(max_tweet_word_count,))
    x1 = Embedding(vocab_size, model_mat_word[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat_word, trainable=False)(input)
    #x1 = Flatten()(x1)
    #x1 = Model(inputs=input, outputs=x1)

    # glove
    #input2 = Input(shape=(max_tweet_word_count,))
    x2 = Embedding(vocab_size, model_mat_glove[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat_glove, trainable=False)(input)
    #x2 = Flatten()(x2)
    #x2 = Model(inputs=input, outputs=x2)

    # skip_gram
    #input3 = Input(shape=(max_tweet_word_count,))
    x3 = Embedding(vocab_size, model_mat_skip_gram[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat_skip_gram, trainable=False)(input)
    #x3 = Flatten()(x3)
    #x3 = Model(inputs=input, outputs=x3)

    combined = concatenate([x1, x2, x3])
    z = Conv1D(100, 5, activation='relu')(combined)
    z = Conv1D(100, 5, activation='relu')(z)
    z = MaxPooling1D()(z)

    z = Conv1D(160, 5, activation='relu')(z)
    z = Conv1D(160, 5, activation='relu')(z)
    z = GlobalMaxPooling1D()(z)
    z = Dropout(0.5)(z)

    #f = Flatten()(z)
    # conv
    #polling
    #flatten
    #dense
    # (evntl. residual conv)
    #z = Dense(10, activation="relu")(z)
    z = Dense(1, activation="sigmoid")(z)
    # ca 20kk total params
    model = Model(inputs=[input], outputs=z)

    # run_opts = tensorflow.RunOptions(report_tensor_allocations_upon_oom=True)
    # compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])  # , options=run_opts)
    # experiment optimizer (adam vs rmsprop)
    # expeniment activation function (liki_relu, elu)
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit([x_train], y_train, epochs=50, verbose=1, batch_size=8000, validation_split=0.1)
    # todo: use return value
    # evaluate the model
    loss, accuracy = model.evaluate([x_test], y_test, verbose=1, batch_size=8000)
    print('Accuracy: %f' % (accuracy * 100))
    # small: Accuracy: 93.041664
    # all data - Accuracy: 79.458750
    #3 embedding Layers:

    model.save('models/sentiment_conv_ep100.h5')

    print('finished')
