## Brief Summary
- I have added README.md that contains steps to setup and run the project.
- I have implement FedAvg and FedSGD Federated Learning technique on a synthetic data for a regression problem using 2 Layers Vanilla Neural Netowrk (with some default settings) that highlights the properites of these algorithms.
- I have ran various experiments and solved multiple practice problems.
- I have attached all the files in this repo, but please use the README to identify the most relevant files that you need to run.

# Detailed Report
1. **DatasetPreparations**: 
    - Used dataset from a file on each node based on row number where row % rank_of_the_process == 0
    - Synthetic Data is polynomial in nature with some white guassian noise with a constant mean and standard deviation. Data has been prepared in a way so that it can illustrate the benefits of fedderated learning even on very short samples and tiny model.
    - I have used entire data as test data on each process to maintain consistency in evaluation.
    - I have plotted the data distribution and put it in images folder for your reference.
    - **Privacy Preserving Feature**: Since each node has it's own training data, data is not shared between the systems which preservies privacy. Here I have simulated this using the a file reading based on row ids, but it illustrates the privacy idea.
2. **Training**: 
- In order to benchmark the performance, I implemented the same aarchitecture (that I used in federated learning) neural network (NN) with any distributed federated learning to get a baseline.
- Developed multiple variants of FedAvg method by changing the point of aggregation in various experiments. The global aggregation turned out to be best on this specific data and model architecture, which is might not be true in other cases.
- Also, implemented the FedSGD technique where we were doing the aggregations at every epoch of the training.
- Trained the models in a way that the results are reproducible.
3. **Metrics**:
- For the given data and model, FedSGD was closer to the baseline than FedAvg technique.
- For Baseline Model:
    - Final loss: 1141.0064697265625
    - R-squared: 0.7502739980476314
    - Mean Absolute Error: 27.496908950417136
- For FedAvg:
    - After Reducing at Rank 0, Final loss: 1318.7928466796875
    - After Reducing at Rank 0, R-squared: 0.7113628476964176
    - After Reducing at Rank 0, Mean Absolute Error: 28.933150488261308
- For FedSGD:
    - After Reducing at Rank 2, Final loss: 1104.7113037109375
    - After Reducing at Rank 2, R-squared: 0.7582177084580953
    - After Reducing at Rank 2, Mean Absolute Error: 27.069782214178403
4. **Distributed System Perspective Analysis**:
- **Message Size**: 
    - From total number of messages/parameters perspective per synchronization, the number of messages remain same, but since FedSGD synchronizes more frequently than FedAvg, overall it sends more number of messages in total.
    - Moreover, in practice, since storing gradients require higher floating precision points than parameters, FedSGD has even more requirements in terms of overall message size capacity than FedAvg.
- **Communication**: 
    - FedSGD requires more bandwidth than FedAvg as it synchronizes after every gradient computation while FedAvg runs after a few epochs. 
- **Deadlock**: I have used Reduceall API of MPI which is a blocking API and since there is no interdependency anywhere else, deadlock will not occur in this scenario.

6. **Raw Outputs**:

```bash
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning % python data_generator.py
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning %
```

```bash
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning % python baseline.py
Epoch 100, Loss: 2089.2651
Epoch 200, Loss: 1303.1350
Epoch 300, Loss: 1183.1193
Epoch 400, Loss: 855.4464
Epoch 500, Loss: 1151.4851
Final loss: 1141.0064697265625
R-squared: 0.7502739980476314
Mean Absolute Error: 27.496908950417136
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning %
```

```bash
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning % mpiexec -n 3 python -m mpi4py fedavg.py
Rank 1, Epoch 100, Loss: 2110.9556
Rank 2, Epoch 100, Loss: 3536.0410
Rank 0, Epoch 100, Loss: 10280.8613
Rank 1, Epoch 200, Loss: 2538.8777
Rank 2, Epoch 200, Loss: 4212.5610
Rank 0, Epoch 200, Loss: 2463.8997
Rank 1, Epoch 300, Loss: 1612.2982
Rank 2, Epoch 300, Loss: 2494.3889
Rank 0, Epoch 300, Loss: 1469.9310
Rank 1, Epoch 400, Loss: 1643.7924
Rank 2, Epoch 400, Loss: 1476.4232
Rank 0, Epoch 400, Loss: 1010.8104
Rank 1, Epoch 500, Loss: 2468.4641
Rank 2, Epoch 500, Loss: 2134.3157
Rank 0, Epoch 500, Loss: 616.8036
Rank 1, Epoch 600, Loss: 2267.4187
Rank 2, Epoch 600, Loss: 2237.0171
Rank 0, Epoch 600, Loss: 1052.4215
Rank 1, Epoch 700, Loss: 746.2194
Rank 2, Epoch 700, Loss: 277.5680
Rank 0, Epoch 700, Loss: 1110.7329
Rank 1, Epoch 800, Loss: 789.4324
Rank 2, Epoch 800, Loss: 370.7536
Rank 0, Epoch 800, Loss: 1323.3073
Rank 1, Epoch 900, Loss: 689.0551
Rank 2, Epoch 900, Loss: 422.8376
Rank 0, Epoch 900, Loss: 953.7806
Rank 1, Epoch 1000, Loss: 1693.9812
Rank 1, Final loss: 2973.50830078125
Rank 1, R-squared: 0.3492040410369186
Rank 1, Mean Absolute Error: 41.55154439983188
Rank 2, Epoch 1000, Loss: 712.4804
Rank 2, Final loss: 1170.5313720703125
Rank 2, R-squared: 0.7438120136705788
Rank 2, Mean Absolute Error: 27.957019440510717
Rank 0, Epoch 1000, Loss: 1276.2207
Rank 0, Final loss: 1133.345703125
Rank 0, R-squared: 0.7519506798547794
Rank 0, Mean Absolute Error: 27.518016614689017
After Reducing at Rank 0, Final loss: 1318.7928466796875
After Reducing at Rank 0, R-squared: 0.7113628476964176
After Reducing at Rank 0, Mean Absolute Error: 28.933150488261308
After Reducing at Rank 1, Final loss: 1318.7928466796875
After Reducing at Rank 1, R-squared: 0.7113628476964176
After Reducing at Rank 1, Mean Absolute Error: 28.933150488261308
After Reducing at Rank 2, Final loss: 1318.7928466796875
After Reducing at Rank 2, R-squared: 0.7113628476964176
After Reducing at Rank 2, Mean Absolute Error: 28.933150488261308
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning %
```

```bash
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning % mpiexec -n 3 python -m mpi4py fedsgd.py
Rank 0, Epoch 100, Loss: 12272.5566
Rank 1, Epoch 100, Loss: 2060.7283
Rank 2, Epoch 100, Loss: 3466.8164
Rank 0, Epoch 200, Loss: 13607.0215
Rank 1, Epoch 200, Loss: 2272.1980
Rank 2, Epoch 200, Loss: 3691.0137
Rank 0, Epoch 300, Loss: 6530.2168
Rank 1, Epoch 300, Loss: 1187.6240
Rank 2, Epoch 300, Loss: 1649.3684
Rank 0, Epoch 400, Loss: 1473.7737
Rank 1, Epoch 400, Loss: 1192.6494
Rank 2, Epoch 400, Loss: 524.2543
Rank 0, Epoch 500, Loss: 999.8586
Rank 1, Epoch 500, Loss: 2025.6960
Rank 2, Epoch 500, Loss: 673.4999
Rank 0, Epoch 600, Loss: 1411.2216
Rank 1, Epoch 600, Loss: 2137.0020
Rank 2, Epoch 600, Loss: 787.1365
Rank 0, Epoch 700, Loss: 967.8832
Rank 1, Epoch 700, Loss: 2164.3813
Rank 2, Epoch 700, Loss: 450.4697
Rank 0, Epoch 800, Loss: 1318.3406
Rank 1, Epoch 800, Loss: 1234.2527
Rank 2, Epoch 800, Loss: 395.4550
Rank 0, Epoch 900, Loss: 1081.6978
Rank 1, Epoch 900, Loss: 2243.7886
Rank 2, Epoch 900, Loss: 453.3144
Rank 0, Epoch 1000, Loss: 1168.5691
Rank 1, Epoch 1000, Loss: 1497.8383
Rank 2, Epoch 1000, Loss: 802.7423
Rank 0, Final loss: 1104.7113037109375
Rank 0, R-squared: 0.7582177084580953
Rank 0, Mean Absolute Error: 27.069782214178403
Rank 1, Final loss: 1104.7113037109375
Rank 1, R-squared: 0.7582177084580953
Rank 1, Mean Absolute Error: 27.069782214178403
Rank 2, Final loss: 1104.7113037109375
Rank 2, R-squared: 0.7582177084580953
Rank 2, Mean Absolute Error: 27.069782214178403
(mpi4py) (base) therajmaurya@Rajs-MacBook-Pro mpi4py-federated-learning %
```
