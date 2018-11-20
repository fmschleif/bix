from __future__ import division

import datetime
import glob
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from Study import Study


class CrossValidation(Study):

    def __init__(self, clfs, streams=None, test_size=1, path="study", param_path="search", max_samples=1000000):
        super().__init__(streams=streams, path=path)

        if type(clfs) == 'list':
            raise ValueError("Must be classifer list")

        self.non_multiflow_metrics = ["time","mean_std","window_std"]
        self.clfs = clfs
        self.param_path = param_path
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
            self.chwd_root()
            os.chdir(os.path.join(os.getcwd(),self.path))
            for stream in self.streams:
                print(clf.__class__.__name__)
                try: clf.reset()
                except NotImplementedError: clf.__init__()
                clf = self.set_clf_params(clf, params, stream.name)
                local_result = []
                for i in range(self.test_size):
                    stream.prepare_for_use()
                    path_to_save = clf.__class__.__name__+"_performance_on_"+stream.name+"_"+self.date+".csv"
                    evaluator = EvaluatePrequential(
                        show_plot=False, max_samples=self.max_samples, restart_stream=True, batch_size=10, metrics=self.metrics,output_file=path_to_save)
                    evaluator.evaluate(stream=stream, model=clf)

                    output= np.array([[m for m in evaluator._data_buffer.data[n]["mean"]] for n in evaluator._data_buffer.data]+[
                                        [evaluator.running_time_measurements[0]._total_time]]).T.flatten().tolist()+np.std(pd.read_csv(path_to_save,comment='#',header=0).values[:,1:3],axis=0).tolist()
                    print(path_to_save+" "+str(output))
                    local_result.append(output)

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
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(),self.param_path))
        try:   
            files = glob.glob("Best_runs*"+clf.__class__.__name__+"*.csv")

            file = self.determine_newest_file(files)
            return pd.read_csv(files[0]) if len(file) > 0 else []
        except FileNotFoundError:
            return None

    def save_summary(self):
        self.chwd_root()
        os.chdir(os.path.join(os.getcwd(),self.path))
        for i, metric in enumerate(self.metrics+self.non_multiflow_metrics):
            values = np.array([elem[1:]
                               for elem in self.result[-1]])[:, :, i].tolist()
            names = [[elem[0]] for elem in self.result[-1]]
            df = pd.DataFrame([n+elem for n, elem in zip(names, values)],
                              columns=["Classifier"]+[s.name for s in self.streams])
            df = df.round(3)
            df.to_csv(path_or_buf="_CV_Study"+"_"+metric+"_"+self.date +
                      "_N_Classifier_"+str(len(self.clfs))+".csv", index=False)

    def determine_newest_file(self, files):
        dates = [datetime.strptime(re.search(
            r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}', file).group(), self.date_format) for file in files]
        return files[dates.index(max(dates))] if len(dates) > 0 else []
