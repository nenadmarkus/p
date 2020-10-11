import os
import ctypes
import numpy
import time

os.system("cc tracer.c -fPIC -O3 -shared -o tracer.so")
lib = ctypes.cdll.LoadLibrary('./tracer.so')
os.system('rm tracer.so')

#
#
#

eps = 0.0001

nrows = 8*512
ncols = 8*512

intfield = numpy.zeros((nrows, nrows, 3), dtype=numpy.float32)
normals = numpy.zeros((nrows, nrows, 3), dtype=numpy.float32)
mindistfield = numpy.zeros((nrows, nrows), dtype=numpy.float32)

t = time.time()

lib.compute(
	ctypes.c_int(nrows),
	ctypes.c_int(ncols),
	ctypes.c_void_p(intfield.ctypes.data),
	ctypes.c_void_p(mindistfield.ctypes.data),
	ctypes.c_void_p(normals.ctypes.data),
	ctypes.c_float(eps)
)

print("* elapsed time: %d [ms]" % int(1000.0*(time.time() - t)))

#
#
#

L, O, H = 8, 3, 24
results = intfield.reshape( (nrows*ncols, 3) ).copy()
#results = normals.reshape((nrows*ncols, 3))
#results = numpy.concatenate(
#	(intfield.reshape((nrows*ncols, 3)), normals.reshape((nrows*ncols, 3))),
#	1
#)
for i in range(0, L):
	if i==L-1:
		W = numpy.random.randn(results.shape[1], O)
	else:
		W = numpy.random.randn(results.shape[1], H)
	#results = numpy.clip(numpy.matmul(results, W), -1.0, +1.0)
	results = numpy.tanh(numpy.matmul(results, W))
results = (1 + results)/2.0

rgb = (255.0*results.reshape(nrows, ncols, results.shape[-1])).astype(numpy.uint8)
rgb[:, :, 0][mindistfield > eps] = 255
rgb[:, :, 1][mindistfield > eps] = 255
rgb[:, :, 2][mindistfield > eps] = 255

import cv2
rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
cv2.imwrite("rez.png", rgb)
#cv2.imshow("...", rgb)
#cv2.imshow("....", mindistfield)
#cv2.waitKey(0)
