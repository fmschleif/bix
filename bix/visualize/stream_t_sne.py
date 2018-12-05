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
from bix.utils.drift_tsne import DriftTSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

class T_sne_stream_visualizer():
    """t-distributed stochastical neighbour embedding
    Visualizes a stream splitted in steps divided by a concept drift detector.
    Thus, each figure should show a new data distribution.
    Parameters
    ----------
    stream : Instance of skmultiflow.data.base_stream
            Stream which will be visualized
    path :   string
            Relative or absolute path which represent the path where figures are
            saved to
    drift_handling : string, ('KS' or 'ADWIN')
            Indicates which drift detector will be used for splitting the batches
    confidence : float
            Confidence used by the drift handling algorithm
    """
    # TODO: implement Z-trans to normalize each batch
    def __init__(self, stream, path=None, drift_handling='KS', confidence=0.001):
        if not isinstance(stream, Stream):
            raise TypeError('Wrong type, expected Stream, but was ', type(stream))
        else:
            self.stream = stream
        
        self.path = path
        self.confidence = confidence
        
        if (drift_handling == 'KS' or drift_handling=='ADWIN'):
            self.drift_handling = drift_handling
        else: 
            raise TypeError('Wrong type of drift handling: {}'.format(drift_handling))
    
    def visualize(self):
        """Starts the visualization of the stream and saves it to the given path"""
        stream = self.stream
        
        if self.path is not None:
            if (os.path.isabs(self.path)):
                path = self.path
            else:
                path = os.path.join(os.getcwd(), self.path)
        else: 
            path = os.path.join(os.getcwd(), 'figure', 
                                datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            if not os.path.exists('figure'):
                os.mkdir('figure')
            
        if not os.path.exists(path):
            os.mkdir(path)
        
        ############################################################
        #                                                          #
        #                       DATA SETUP                         #
        #                                                          #
        ############################################################
    
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
        
        drift_tsne = DriftTSNE(drift_handling=self.drift_handling, confidence=self.confidence)
        
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
                self._plot_batch(X_batch=complete_X, y_batch=complete_y, detections=detections, path=path)
                init = True
                
        print(50 * '-')
        print('Finished Drift Detection')
        
    def _plot_batch(self, X_batch, y_batch, detections, path):
        """Internal Method"""
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

if __name__ == '__main__':
    visualizer = T_sne_stream_visualizer(ConceptDriftStream(MIXEDGenerator(classification_function=0), 
                            MIXEDGenerator(classification_function=1),
                            width=1,
                            position=1500),
                path=None,
                drift_handling='KS',
                confidence=0.001)
    visualizer.visualize()
        