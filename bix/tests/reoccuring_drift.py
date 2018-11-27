

from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from bix.utils.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.lazy.knn import KNN

def reoccuring_test():
    s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
    s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)

    """1. Create stream"""
    stream = ReoccuringDriftStream(stream=s1,
                            drift_stream=s2,
                            random_state=None,
                            alpha=90.0, # angle of change grade 0 - 90
                            position=2000,
                            width=500)
    
    stream.prepare_for_use()

    oza = OzaBaggingAdwin(base_estimator=KNN())

    """3. Setup evaluator"""
    evaluator = EvaluatePrequential(show_plot=True,batch_size=10,
                                    max_samples=5000,
                                    metrics=['accuracy', 'kappa_t', 'kappa_m', 'kappa'],    
                                    output_file=None)

    """4. Run evaluator"""
    evaluator.evaluate(stream=stream, model=oza)

if __name__ == "__main__":
    reoccuring_test()
    
  