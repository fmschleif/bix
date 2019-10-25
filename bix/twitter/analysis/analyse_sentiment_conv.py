import sys

import tensorflow
from keras import Sequential, Input, Model
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Embedding, Flatten, Dense, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from bix.twitter.base.utils import load_model_mat, load_training_sentiment_data_small, load_pickle, load_csv, \
    load_training_sentiment_data
from bix.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer


def train_model_convolutional(x, y, embedding_mats):
    #    print(device_lib.list_local_devices())
    #    exit(0)
    padded_x = x

    max_tweet_word_count = load_pickle('max_tweet_word_count.pickle')
    tok = load_tokenizer()
    vocab_size = tok.num_words

    #if 'test_all_data' in args:
    #    padded_x = load_pickle('tokenized/learn/padded_x.pickle')
    #    y = load_csv('learn/lables.csv')

    enc_y = np_utils.to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(padded_x, enc_y, test_size=0.2, random_state=5)

    #model_mat_word = load_model_mat('embedding_word')
    #print(f'model_mat_word.shape: {model_mat_word[0].shape}')
    #model_mat_glove = load_model_mat('embedding_glove')
    #print(f'model_mat_glove.shape: {model_mat_glove[0].shape}')
    #model_mat_skip_gram = load_model_mat('embedding_skip_gram')
    #print(f'model_mat_skip_gram.shape: {model_mat_skip_gram[0].shape}')
    # model_mat_skip_gram = load_model_mat('embedding_skip_gram')

    # vocab_size = len(tok.word_index) + 1

    # model = Sequential()
    # model.add(Embedding(vocab_size, model_mat_word[0].shape[1], input_length=max_tweet_word_count,
    #                    weights=model_mat_word))
    # model.add(Flatten())

    # model.add(Dense(1, activation='sigmoid'))

    # word
    # input = Input(shape=(max_tweet_word_count,))
    xs = []
    for mat in embedding_mats:
        xs += Embedding(vocab_size, mat[0].shape[1], input_length=max_tweet_word_count,
                       weights=mat, trainable=False)(input)

    combined = concatenate(xs)
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
    z = Dense(len(y_test[0]), activation="softmax")(z)
    # ca 20kk total params
    model = Model(inputs=[input], outputs=z)

    # run_opts = tensorflow.RunOptions(report_tensor_allocations_upon_oom=True)
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])  # , options=run_opts)
    # experiment optimizer (adam vs rmsprop)
    # expeniment activation function (liki_relu, elu)
    # summarize the model
    print(model.summary())
    # fit the model

    es = EarlyStopping(monitor='val_loss')
    model.fit([x_train], y_train, epochs=50, verbose=1, batch_size=8000, validation_split=0.1, callbacks=[es])
    # todo: use return value
    # evaluate the model
    loss, accuracy = model.evaluate([x_test], y_test, verbose=1, batch_size=8000)
    print('Accuracy: %f' % (accuracy * 100))
    # small: Accuracy: 93.041664
    # all data - Accuracy: 79.458750
    #3 embedding Layers:

    model.save('sentiment_conv_ep100.h5')

    print('finished')


def predict_model_convolutional(x, model: Model = None, evaluate: bool=False, y=None):
    #    print(device_lib.list_local_devices())
    #    exit(0)
    padded_x = x

    if model is None:
        model = load_model('sentiment_conv_ep100.h5')

    enc_y = np_utils.to_categorical(y)

    print(model.summary())

    predictions = model.predict([x], verbose=1)
    # evaluate the model
    if evaluate:
        loss, accuracy = model.evaluate([x], enc_y, verbose=1)
        print('Accuracy: %f' % (accuracy * 100))

    return predictions


