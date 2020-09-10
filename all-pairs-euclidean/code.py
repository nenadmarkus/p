import torch
import numpy
import time

#
#
#

def all_pairs_euclid_naive(A, B):
	#
	D = numpy.zeros((A.shape[0], B.shape[0]), dtype=numpy.float32)
	for i in range(0, D.shape[0]):
		for j in range(0, D.shape[1]):
			D[i, j] = numpy.linalg.norm(A[i, :] - B[j, :])
	#
	return D

def all_pairs_euclid_numpy(A, B):
	#
	sqrA = numpy.broadcast_to(numpy.sum(numpy.power(A, 2), 1).reshape(A.shape[0], 1), (A.shape[0], B.shape[0]))
	sqrB = numpy.broadcast_to(numpy.sum(numpy.power(B, 2), 1).reshape(B.shape[0], 1), (B.shape[0], A.shape[0])).transpose()
	#
	return numpy.sqrt(
		sqrA - 2*numpy.matmul(A, B.transpose()) + sqrB
	)

def all_pairs_euclid_torch(A, B):
	#
	sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
	sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
	#
	return torch.sqrt(
		sqrA - 2*torch.mm(A, B.t()) + sqrB
	)

#
#
#

M, N, d = 2048, 2048, 1024
A = numpy.random.randn(M, d).astype(numpy.float32)
B = numpy.random.randn(N, d).astype(numpy.float32)

#
all_pairs_euclid_naive(A, B) # warmup run
t = time.time()
D_naive = all_pairs_euclid_naive(A, B)
print('* elapsed time (naive): %d [ms]' % int(1000*(time.time() - t)))

#
all_pairs_euclid_numpy(A, B) # warmup run
t = time.time()
D_numpy = all_pairs_euclid_numpy(A, B)
print('* elapsed time (numpy): %d [ms]' % int(1000*(time.time() - t)))

#
A = torch.from_numpy(A)
B = torch.from_numpy(B)
all_pairs_euclid_torch(A, B) # warmup run
t = time.time()
D_torch = all_pairs_euclid_torch(A, B)
print('* elapsed time (torch): %d [ms]' % int(1000*(time.time() - t)))
D_torch = D_torch.numpy()

if torch.cuda.is_available() is True:
	all_pairs_euclid_torch(A.cuda(), B.cuda()).cpu()
	t = time.time()
	D_torch_cuda = all_pairs_euclid_torch(A.cuda(), B.cuda()).cpu()
	print('* elapsed time (torch): %d [ms]' % int(1000*(time.time() - t)))

#
print( numpy.max( D_torch - D_naive ) )
print( numpy.max( D_numpy - D_naive ) )