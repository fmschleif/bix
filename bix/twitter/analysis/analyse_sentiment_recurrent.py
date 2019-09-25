import sys

import tensorflow
from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Dense, concatenate, Dropout, LSTM
from sklearn.model_selection import train_test_split

from bix.twitter.base.utils import load_model_mat, load_training_sentiment_data_small, load_training_sentiment_data

if __name__ == '__main__':
    #    print(device_lib.list_local_devices())
    #    exit(0)

    args = sys.argv

    tok, y, padded_x, unpadded_x, max_tweet_word_count, vocab_size = load_training_sentiment_data()

    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.50) # was 0.10

    model_mat = None
    if 'word' in args:
        model_mat = load_model_mat('embedding_word')
    if 'glove' in args:
        model_mat = load_model_mat('embedding_glove')
    if 'skip_gram' in args:
        model_mat = load_model_mat('embedding_skip_gram')


    print(f'model_mat.shape: {model_mat[0].shape}')
    #model_mat_glove = load_model_mat('embedding_glove')
    #print(f'model_mat_glove.shape: {model_mat_glove[0].shape}')
    # model_mat_skip_gram = load_model_mat('embedding_skip_gram')

    # vocab_size = len(tok.word_index) + 1

    # model = Sequential()
    # model.add(Embedding(vocab_size, model_mat_word[0].shape[1], input_length=max_tweet_word_count,
    #                    weights=model_mat_word))
    # model.add(Flatten())

    # model.add(Dense(1, activation='sigmoid'))

    # word
    embedding_layer = Embedding(vocab_size, model_mat[0].shape[1], input_length=max_tweet_word_count,
                   weights=model_mat, trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    #x1 = Flatten()(x1)
    #x1 = Model(inputs=input1, outputs=x1)

    # glove
    #input2 = Input(shape=(max_tweet_word_count,))
    #x2 = Embedding(vocab_size, model_mat_glove[0].shape[1], input_length=max_tweet_word_count,
    #               weights=model_mat_glove, trainable=False)(input2)
    #x2 = Flatten()(x2)
    #x2 = Model(inputs=input2, outputs=x2)

    #combined = concatenate([x1.output, x2.output])

    #z = Dense(10, activation="relu")(x1)
    #z = Dense(1, activation="sigmoid")(x1)

    #model = Model(inputs=input1, outputs=z)

    # run_opts = tensorflow.RunOptions(report_tensor_allocations_upon_oom=True)
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # , options=run_opts)
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(x_train, y_train, epochs=2, verbose=1, batch_size=8000, validation_split=0.1)
    # evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1, batch_size=8000)
    print('Accuracy: %f' % (accuracy * 100))
    #all:
    # Accuracy: 74.407125 (2 epocs)

    print('finished')
