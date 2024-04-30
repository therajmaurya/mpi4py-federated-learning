# Guide to run the components of this project:

1. To generate the dataset
```bash
$ python data_generator.py
```

2. To run the baseline model
```bash
$ python baseline.py
```

3. To run the Distributed FedAvg Model:
```bash
$ mpiexec -n 3 python -m mpi4py fedavg.py
```

3. To run the Distributed FedSGD Model:
```bash
$ mpiexec -n 3 python -m mpi4py fedsgd.py
```

# Guide to set-up the system:

1. Create and activate a conda environment with python=3.10
```
$ conda create -n mpi4py python=3.10
$ conda activate mpi4py
```

2. Run following command:
```
$ conda install -c conda-forge mpi4py openmpi
```

3. To execute the files (not-preferred):
```
$ mpiexec -n 3 script.py 
```

3.1 To excute the files with exceptions that might cause deadlock (https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html) - preferred option:
```
mpiexec -n 3 python -m mpi4py script.py
```

Links:
- https://mpi4py.readthedocs.io/en/latest/install.html
- https://mpi4py.readthedocs.io/en/stable/tutorial.html
