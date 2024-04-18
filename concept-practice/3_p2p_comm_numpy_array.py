from mpi4py import MPI
import numpy

# help(MPI)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# help(comm)

# passing MPI datatypes explicitly
if rank == 0:
    data = numpy.arange(1000, dtype=int)
    comm.Send([data, MPI.INT], dest=1, tag=77)
    print(f"I am {rank}. I sent data with dtype: {data}")
elif rank == 1:
    data = numpy.empty(1000, dtype=int)
    comm.Recv([data, MPI.INT], source=0, tag=77)
    print(f"I am {rank}. I received data with dtype: {data}")
# automatic MPI datatype discovery
if rank == 0:
    data = numpy.arange(100, dtype=numpy.float64)
    comm.Send(data, dest=1, tag=13)
    print(f"I am {rank}. I sent data WITHOUT dtype: {data}")
elif rank == 1:
    data = numpy.empty(100, dtype=numpy.float64)
    comm.Recv(data, source=0, tag=13)
    print(f"I am {rank}. I received data WITHOUT dtype: {data}")
