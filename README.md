# mpi4py-federated-learning

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
