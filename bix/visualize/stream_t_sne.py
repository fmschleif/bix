#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:34:49 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from sklearn.manifold import TSNE
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.base_stream import Stream
from starting_point import DriftTSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

############################################################
#                                                          #
#                       t-SNE                              #    
#                                                          #
############################################################
def _plot_batch(X_batch, y_batch, detections, path):
    plt.ioff() # interactive shell off
    i = len(detections) - 1
    
    print(50 * '-')
    print('Start t-SNE Iteration ', i+1)
                  
    # Optimize perplexity between 5 and 50
    perplex=5
    min_kl = 1
    
    while perplex <= 50:
        t_sne = TSNE(perplexity=perplex, n_iter=1000)
        X_embedding = t_sne.fit_transform(X=X_batch, y=y_batch)
        
        if (t_sne.kl_divergence_ < min_kl):
            opt_perplex = perplex
            min_kl = t_sne.kl_divergence_
            
        progress = int(perplex / 50 * 100)
        print('Progress: {} %'.format(progress))
        perplex += 5

        
    t_sne = TSNE(perplexity=opt_perplex, n_iter=1000)
    X_embedding = t_sne.fit_transform(X=X_batch, y=y_batch)
    
    plt.title('Batch: {} Position: {} Size: {}'.format(i+1, detections[i] + 1, X_batch.shape[0]))
    plt.ylim([np.min(X_embedding.T[1]) - 10, np.max(X_embedding.T[1]) + 10])
    plt.xlim([np.min(X_embedding.T[0]) - 10, np.max(X_embedding.T[0]) + 10])
    fig = plt.scatter(x=X_embedding.T[0], y=X_embedding.T[1], c=y_batch)
    plt.savefig(fname='{}/batch{}.png'.format(path,i+1), dpi=350)
    fig.remove()
    plt.clf()
   
    print(50 * '-')
    print('Finished t-SNE Iteration ', i+1)

def t_sne_stream_visualization(stream):
    path = 'figure/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists('figure'):
        os.mkdir('figure')
    os.mkdir(path)
    
    ############################################################
    #                                                          #
    #                       DATA SETUP                         #
    #                                                          #
    ############################################################
    if not isinstance(stream, Stream):
        raise TypeError('Wrong type, expected Stream, but was ', type(stream))

    stream.prepare_for_use()

    ############################################################
    #                                                          #
    #                       DETECT DRIFT                       #
    #                                                          #
    ############################################################
    print(50 * '-')
    print('Starting Drift Detection')
    detections = []
    init = True
    
    drift_tsne = DriftTSNE(drift_handling='KS', confidence=0.001)
    
    iterations = 3000
    
    for i in range(iterations):
        X, y = stream.next_sample(batch_size=1)
        if init == True:
            complete_X, complete_y = np.array(X), np.array(y)
            init = False
        else:
            complete_X = np.append(complete_X, X, axis=0)
            complete_y = np.append(complete_y, y, axis=0)
     
        drift_detected = drift_tsne.concept_drift_detection(X, y)
        
        if(drift_detected or i >= iterations - 1):
            detections.append(i)
            _plot_batch(X_batch=complete_X, y_batch=complete_y, detections=detections, path=path)
            init = True
            
    print(50 * '-')
    print('Finished Drift Detection')

if __name__ == '__main__':
    t_sne_stream_visualization(ConceptDriftStream(MIXEDGenerator(classification_function=0), 
                            MIXEDGenerator(classification_function=1),
                            width=1,
                            position=1500)) 
        