#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:27:09 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import numpy as np

from rslvq import RSLVQ
from arslvq import RSLVQ as ARSLVQ
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from sklearn.metrics import accuracy_score

from scipy import stats
import scipy.io as sio

def stream_example():
    """Create stream"""
    stream = SEAGenerator(noise_percentage=0.1)
    
    stream.prepare_for_use()

    """Init BRSLVQ"""
    clf = [RSLVQ(sigma=5.0, batch_size=1, n_epochs=1), 
        RSLVQ(sigma=5.0, batch_size=5, n_epochs=1), 
        RSLVQ(sigma=5.0, batch_size=10, n_epochs=1)]

    """Evaluate"""
    evaluator = EvaluatePrequential(max_samples=10000, batch_size=100,
                                    show_plot=True)

    """Start evaluation"""
    evaluator.evaluate(stream=stream, model=clf, model_names=['bs=1', 'bs=5', 'bs=10'])

def batch_example():

    """Load in reuters and image dataset and convert it to numpy arrays"""
    data = sio.loadmat('org_vs_people_1_full.mat')
    print(data)
    Xs = np.array(data["Xs"])
    Xt = np.array(data["Xt"])
    Ys = np.array(data["Ys"])
    Yt = np.array(data["Yt"])

    Ys = Ys.reshape(Ys.shape[0],)
    Yt = Yt.reshape(Yt.shape[0],)
    print(Xs.shape)

    """Preprocessing"""
    Xs = stats.zscore(Xs.T,axis=None)
    Xt = stats.zscore(Xt.T,axis=None)
    print(Xs.shape)

    """Normal RSLVQ without transfer learning"""
    print("Start Training...")
    rslvq = RSLVQ(prototypes_per_class=10, batch_size=10, sigma=0.5)
    rslvq.fit(Xs, Ys)
    print("\r Start Prediction... ")
    prediction = rslvq.predict(Xt)
    print("\r Run finished!")
    print('Accuracy is {}'.format(accuracy_score(prediction, Yt)))
    print(prediction)

if __name__ == "__main__":

    batch_example()
