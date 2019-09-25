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

    # if 'test_all_data' in args:
    #    padded_x = load_pickle('tokenized/learn/padded_x.pickle')
    #    y = load_csv('learn/lables.csv')

    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.2, random_state=5)
    model_mat = None
    if 'word' in args:
        model_mat = load_model_mat('embedding_word')
    if 'glove' in args:
        model_mat = load_model_mat('embedding_glove')
    if 'skip_gram' in args:
        model_mat = load_model_mat('embedding_skip_gram')
    print(f'model_mat_skip_gram.shape: {model_mat[0].shape}')

    # vocab_size = len(tok.word_index) + 1

    # model = Sequential()
    # model.add(Embedding(vocab_size, model_mat_word[0].shape[1], input_length=max_tweet_word_count,
    #                    weights=model_mat_word))
    # model.add(Flatten())

    # model.add(Dense(1, activation='sigmoid'))

    # word
    input = Input(shape=(max_tweet_word_count,))
    x1 = Embedding(vocab_size, model_mat[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat, trainable=False)(input)
    # x1 = Flatten()(x1)
    # x1 = Model(inputs=input, outputs=x1)
    # x3 = Flatten()(x3)
    # x3 = Model(inputs=input, outputs=x3)

    z = Conv1D(100, 5, activation='relu')(x1)
    z = Conv1D(100, 5, activation='relu')(z)
    z = MaxPooling1D()(z)

    z = Conv1D(160, 5, activation='relu')(z)
    z = Conv1D(160, 5, activation='relu')(z)
    z = GlobalMaxPooling1D()(z)
    z = Dropout(0.5)(z)

    # f = Flatten()(z)
    # conv
    # polling
    # flatten
    # dense
    # (evntl. residual conv)
    # z = Dense(10, activation="relu")(z)
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

    model.save(f'models/sentiment_conv_ep100_{args[1]}.h5')

    print('finished')
