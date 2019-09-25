from keras import Model
from keras.engine.saving import load_model

from bix.twitter.base.utils import load_pickle
import numpy as np

model: Model = load_model('models/sentiment_conv_ep100.h5')
love = load_pickle('preprocessed/hashtag_love.pickle')
sad = load_pickle('preprocessed/hashtag_sad.pickle')
res_love = model.predict_on_batch(love)
res_love = [e[0] for e in res_love]
res_sad = model.predict_on_batch(sad)
res_sad = [e[0] for e in res_sad]


print(f"love - mean: {np.mean(res_love)}, avg: {np.average(res_love)}, acc: {np.average([round(e) for e in res_love])}")
print(f"sad - mean: {np.mean(res_sad)}, avg: {np.average(res_sad)}, acc: {np.average([round(e) for e in res_sad])}")
