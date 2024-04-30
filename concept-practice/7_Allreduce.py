# MPI.Allreduce function is a blocking operation

from mpi4py import MPI
import numpy as np

# Create a communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a send buffer
if rank == 0:
    send_buf = np.array([1, 2, 3])
elif rank == 1:
    send_buf = np.array([4, 5, 6])
else:
    send_buf = np.array([7, 8, 9])
# Create a receive buffer
recv_buf = np.array([0, 0, 0])

# Perform the allreduce operation
comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)

# Print the receive buffer
print(recv_buf)
