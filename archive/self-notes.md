# Self-notes

1. [DatasetPreparations]: Use dataset from a file on each node based on index values (row % rank == 0) **[Privacy Preserving Feature]**
    - Prepare the data such that each node in the system gets mostly one kind of data unique to themselves (even odd row idea for data; odd row data should follow polynomial and even row should follow linear but in the range where polynomial is close to linear)
    - Test data should have all types of data sampled from polynomial distribution [combined data can act as test set as well]
2. [TrainingAndEval]: Compute metrics on test data how the model performs for following:
    - A baseline model (same model & config as in federated learning) without federated learning (this will represent the highest aaccuracy that can be achieved when doing global optimisation) - TIME IT.
    - Score local model metrics on each node and also get the global model metric for comparision **[Model Performance Metrics]**
    - Log how much communication overhead was there. Usually since gradients will be in higher precision points than the model weights, communication overhead for FedSGD will be higher than FedAvg **[Communication Efficiency]**
    - Study **deadlock** scenario
