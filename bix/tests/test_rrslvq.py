
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.lazy.knn import KNN
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.core.base import StreamModel
from skmultiflow.core.pipeline import Pipeline
from bix.classifiers.RRSLVQ import RRSLVQ

def rrslvq_test():
    stream = SEAGenerator(classification_function = 2, random_state = 112, balance_classes = False, noise_percentage = 0.28)     
    stream.prepare_for_use()
    rrslvq = RRSLVQ(prototypes_per_class=10,drift_handling="ADWIN_ERROR",sigma=10,confidence=0.1)    
    oza = OzaBaggingAdwin(base_estimator=KNN(), n_estimators=2)

    model_names = ["oza","rrslvq"]
    pipe = Pipeline([('Classifier', oza)])
    classifier = [pipe, rrslvq]

    evaluator = EvaluatePrequential(show_plot=True,max_samples=5000, 
    restart_stream=True,batch_size=10,metrics=['kappa', 'kappa_m', 'accuracy']) 

    evaluator.evaluate(stream=stream, model=classifier,model_names=model_names)




if __name__ == "__main__":
    rrslvq_test()
  
    