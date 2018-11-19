from __future__ import division

import math
import copy
import sys
import glob
import os
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
from bix.preprocessing.ReoccuringDriftStream import ReoccuringDriftStream
from bix.preprocessing.geometric_median import *
#from kswin import KSWIN
#from rrslvq import RRSLVQ as rRSLVQ
#from rslvq import RSLVQ as bRSLVQ
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
from bix.preprocessing.GridSearch import GridSearch
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
from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
#from stream_rslvq import RSLVQ as sRSLVQ
from skmultiflow.lazy.sam_knn import SAMKNN
import itertools
import datetime
from bix.preprocessing.Study import Study
import re
from datetime import datetime
from joblib import Parallel, delayed
import time


class CrossValidation(Study):

    def __init__(self, clfs, streams=None, test_size=1, path="study\\", param_path="search\\", max_samples=1000000):
        super().__init__(streams=streams, path=path)

        if type(clfs) == 'list':
            raise ValueError("Must be classifer list")

        self.clfs = clfs
        self.test_size = test_size
        self.result = [[] for e in self.metrics]
        self.max_samples = max_samples
        self.time_result = []

    def reset(self):
        self.__init__(self, self.clfs)

    def test(self):
        start = time.time()
        self.result.append(Parallel(n_jobs=-1)
                           (delayed(self.clf_job)(clf) for clf in self.clfs))
        end = time.time() - start
        print("\n--------------------------\n")
        print("Duration of cross validation "+str(end)+" seconds")

    def clf_job(self, clf):
            clf_result = []
            time_result = []
            params = self.search_best_parameters(clf)
            for stream in self.streams:
                print(clf.__class__.__name__)
                clf.reset()
                clf = self.set_clf_params(clf, params, stream.name)
                local_result = []
                for i in range(self.test_size):
                    stream.prepare_for_use()
                    evaluator = EvaluatePrequential(
                        show_plot=False, max_samples=self.max_samples, restart_stream=True, batch_size=10, metrics=self.metrics)
                    evaluator.evaluate(stream=stream, model=clf)
                    local_result.append(np.array([[m for m in evaluator._data_buffer.data[n]["mean"]] for n in evaluator._data_buffer.data]+[
                                        [evaluator.running_time_measurements[0]._total_time]]).T.flatten().tolist())
                clf_result.append(np.mean(local_result, axis=0).tolist())

            return [clf.__class__.__name__]+clf_result

    def test_real_world(self):
        self.streams = self.init_real_world()
        self.result.append(self.clf_job(clf) for clf in self.clfs)

    def set_clf_params(self, clf, df, name):
        if isinstance(df, pd.DataFrame):
            row = df[df['Stream'] == name]
            if len(row) == 1:
                for k, v in zip(list(row.keys()), row.values[0]):
                    if k in clf.__dict__.keys():
                        clf.__dict__[k] = int(v) if type(v) == float else v
        return clf

    def search_best_parameters(self, clf):
        try:
            os.chdir("search/")
            files = glob.glob("Best_runs*"+clf.__class__.__name__+"*.csv")

            file = self.determine_newest_file(files)
            return pd.read_csv(files[0]) if len(file) > 0 else []
        except FileNotFoundError:
            return None

    def save_summary(self):
        for i, metric in enumerate(self.metrics+["time"]):
            values = np.array([elem[1:]
                               for elem in self.result[-1]])[:, :, i].tolist()
            names = [[elem[0]] for elem in self.result[-1]]
            df = pd.DataFrame([n+elem for n, elem in zip(names, values)],
                              columns=["Classifier"]+[s.name for s in self.streams])
            df = df.round(3)
            df.to_csv(path_or_buf=self.path+"CV_Study"+"_"+metric+"_"+self.date +
                      "_N_Classifier_"+str(len(self.clfs))+".csv", index=False)

    def determine_newest_file(self, files):
        dates = [datetime.strptime(re.search(
            r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}', file).group(), self.date_format) for file in files]
        return files[dates.index(max(dates))] if len(dates) > 0 else []


if __name__ == "__main__":

    # s1 = SineGenerator(classification_function=0,balance_classes=False,random_state=112)
    # s2 = SineGenerator(classification_function=1,balance_classes=False, random_state=112)
    # ra_sine = ReoccuringDriftStream(stream=s1, drift_stream=s2,alpha=90.0,position=2000,width=1)
    # rg_sine = ReoccuringDriftStream(stream=s1, drift_stream=s2,alpha=90.0,position=2000,width=1000)

    # stagger1 = STAGGERGenerator(classification_function=0,balance_classes=False,random_state=112)
    # stagger2 = STAGGERGenerator(classification_function=1,balance_classes=False,random_state=112)
    # ra_stagger = ReoccuringDriftStream(stream=stagger1, drift_stream=stagger2, random_state=112,alpha=90.0,position=2000,width=1)
    # rg_stagger = ReoccuringDriftStream(stream=stagger1, drift_stream=stagger2, random_state=112,alpha=90.0,position=2000,width=1000)

    # mixed1 = MIXEDGenerator(classification_function=0,random_state=112,balance_classes=False)
    # mixed2 = MIXEDGenerator(classification_function=1,random_state=112,balance_classes=False)
    # ra_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112,alpha=90.0,position=2000,width=1)
    # rg_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112,alpha=90.0,position=2000,width=1000)
    # streams = [rg_stagger]

    # for stream in streams:
    #     clf = SAMKNN()
    #     stream.prepare_for_use()
    #     evaluator = EvaluatePrequential(show_plot=False,batch_size=1,restart_stream=True,
    #                                     max_samples=1000000,
    #                                     metrics=['accuracy', 'kappa_t', 'kappa_m', 'kappa'],
    #                                     output_file=None)

    #     evaluator.evaluate(stream=stream, model=clf,model_names=["sam"])

    cv = CrossValidation(clfs=[OzaBaggingAdwin(
        base_estimator=KNN()), HAT(), AdaptiveRandomForest()])
    cv.test()
    cv.save_summary()