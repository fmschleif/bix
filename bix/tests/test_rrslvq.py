



if __name__ == "__main__":
    
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
    