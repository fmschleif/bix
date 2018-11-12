from __future__ import division

import math
import copy
import sys
from random import random as rnd
sys.path.append('..\\multiflow-rslvq')
sys.path.append('..\\RSLVQ')
sys.path.append('..\\Reactive_RSLVQ')
sys.path.append("..\\stream_utilities")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
from ReoccuringDriftStream import ReoccuringDriftStream 

from kswin import KSWIN
from rrslvq import RRSLVQ as rRSLVQ
from rslvq import RSLVQ as bRSLVQ
from skmultiflow.core.base import StreamModel
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
#Abrupt Concept Drift Generators
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.sine_generator import SineGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from joblib import Parallel, delayed
# Incremental Concept Drift Generators
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
# No Concept Drift Generators
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.lazy.knn import KNN
from skmultiflow.meta import OzaBaggingAdwin
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from stream_rslvq import RSLVQ as sRSLVQ
import itertools
import os
import datetime
import time
from Study import Study

class GridSearch(Study):

    def __init__(self,clf,grid,streams=None,path="search/",max_samples = 50000):
        super().__init__(streams=streams,path=path)
        if clf == None:
            raise ValueError("Classifier Instance must be set!")
        if grid == None or type(grid) != dict:
            raise ValueError("Parameter Grid must be set and be dictionary!")

        self.clf = clf
        self.grid = grid
        self.best_runs = []
        self.path = path
        self.max_samples = max_samples

    def reset(self):
        self.__init__()

    def search(self):
        start = time.time()
        self.best_runs.extend(Parallel(n_jobs=-1)(delayed(self.self_job)(stream,self.clf,self.grid,self.metrics,self.max_samples) for stream in self.streams))
        end = time.time() -start
        print("\n--------------------------\n")
        print("Duration of cross validation "+str(end)+" seconds")

    def self_job(self,stream,clf,grid,metrics,max_samples):
        results = []
        matrix = list(itertools.product(*[list(v) for v in grid.values()]))
        for param_tuple in matrix:
            clf.reset()
            for i,param in enumerate(param_tuple):
                clf.__dict__[list(grid.keys())[i]] = int(param) if param.dtype == 'int32' else param
            stream.prepare_for_use()
            evaluator = EvaluatePrequential(show_plot=False,max_samples=max_samples, restart_stream=True,batch_size=10,metrics=metrics)
            evaluator.evaluate(stream=stream, model=clf)
            results.append(list(param_tuple)+np.array([[m for m in evaluator._data_buffer.data[n]["mean"]] for n in evaluator._data_buffer.data]).T.flatten().tolist())
        s_name = stream.basename if stream.name==None else stream.name
        dfr = pd.DataFrame(results,columns=list(self.grid.keys())+np.array([[*evaluator._data_buffer.data]]).flatten().tolist())
        dfr = dfr.round(3)
        dfr.to_csv(path_or_buf=self.path+"Result_"+"_"+self.date+"_"+s_name+"_"+self.clf.__class__.__name__+".csv")
        print("\n ------------------ \n")
        print("Best run on "+s_name+" with "+" "+self.clf.__class__.__name__+" "+str(dfr.values[dfr["accuracy"].values.argmax()]))
        return [s_name]+[self.clf.__class__.__name__]+dfr.values[dfr["accuracy"].values.argmax()].tolist()

    def save_summary(self):
        df = pd.DataFrame(self.best_runs,columns=["Stream","Classifier"]+list(self.grid.keys())+self.metrics)
        df.to_csv(path_or_buf=self.path+"Best_runs_"+"_"+self.date+"_"+self.clf.__class__.__name__+"_.csv", index=False)
    

if __name__ == "__main__":  


    grid = {"sigma": np.arange(2,21,2), "prototypes_per_class": np.arange(2,9,2)}
    #grid = {"sigma": np.arange(2,3,2), "prototypes_per_class": np.arange(2,3,2)}

    #gs = GridSearch(rRSLVQ(),grid,max_samples=1000)
    gs = GridSearch(rRSLVQ(),grid)
    gs.search()
    gs.save_summary()

    """
    Save for further parallel runs
    best_runs = []
    import glob, os
    os.chdir("search/")
    for file in glob.glob("Result*.csv"):
        print(file)
        df = pd.read_csv(file,index_col=None)
        file = file[25:-11]
        best_runs.append([file]+["RRSLVQ"]+df.values[df["accuracy"].values.argmax()].tolist()[1:])
        df = pd.DataFrame(best_runs).round(3)
    df.to_csv(path_or_buf="search"+"Best_runs_1_"+"_"+str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M"))+"_"+"RRSLVQ"+"_.csv", index=False, header=False)

    """
