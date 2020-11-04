import numpy
import os
import ctypes
import cv2
import time
import sys

if len(sys.argv)!=4 and len(sys.argv)!=5:
	print("* args: <input image path> <number of trees> <tree depth> [output image path]")
	sys.exit()

imgpath = sys.argv[1]
ntrees = int(sys.argv[2])
tdepth = int(sys.argv[3])

if len(sys.argv) == 5:
	outpath = sys.argv[4]
else:
	outpath = None

#
#
#

os.system('cc api.c -O3 -fPIC -shared -o ud3.lib.so')
ud3lib = ctypes.cdll.LoadLibrary('./ud3.lib.so')
os.system('rm ud3.lib.so')

#
# load the image
#

img = cv2.imread(sys.argv[1]).astype(numpy.float32)/255.0

nrows = img.shape[0]
ncols = img.shape[1]
nchns = 3

targets = numpy.ascontiguousarray(img.copy().reshape(-1, nchns))

#
# generate the tree inputs
#

# construct a 2D array in which each row has integers between 0 and nrows-1
rowmat = numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T
# construct a 2D array in which each column has integers between 0 and ncols-1
colmat = numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)

# normalize coordinates
rowmat = rowmat/max(nrows, ncols) - 0.5
colmat = colmat/max(nrows, ncols) - 0.5

# compute random Fourier features
inputs = []
for i in range(0, 64):
	#
	a = 1.0
	b1 = a*numpy.random.randn()
	b2 = a*numpy.random.randn()
	#
	inputs.append(numpy.cos(2*numpy.pi*(b1*rowmat + b2*colmat)))
	inputs.append(numpy.sin(2*numpy.pi*(b1*rowmat + b2*colmat)))

inputs = numpy.stack(inputs).transpose(1, 2, 0).reshape(-1, len(inputs))
inputs = numpy.ascontiguousarray(inputs.astype(numpy.float32))

#
# learn the trees with gradient boosting
#

trees = numpy.zeros(ntrees, dtype=numpy.int64)

start = time.time()

ud3lib.new_ensemble(
	ctypes.c_float(0.25),
	ctypes.c_void_p(trees.ctypes.data), ctypes.c_int(ntrees), ctypes.c_int(tdepth),
	ctypes.c_void_p(targets.ctypes.data), ctypes.c_int(targets.shape[1]),
	ctypes.c_void_p(inputs.ctypes.data), ctypes.c_int(inputs.shape[1]),
	ctypes.c_void_p(0),
	ctypes.c_int(inputs.shape[0]),
	ctypes.c_int(32)
)

print('* elapsed time (learning): %d [s]' % int(time.time() - start))

#
# compute the approximation
#

predictions = numpy.zeros(targets.shape, dtype=numpy.float32)

start = time.time()

ud3lib.run_ensemble(
	ctypes.c_void_p(trees.ctypes.data), ctypes.c_int(ntrees),
	ctypes.c_void_p(inputs.ctypes.data), ctypes.c_int(inputs.shape[1]),
	ctypes.c_void_p(predictions.ctypes.data), ctypes.c_int(predictions.shape[1]),
	ctypes.c_int(inputs.shape[0])
)

print('* elapsed time (prediction): %d [s]' % int(time.time() - start))

ud3lib.del_ensemble(
	ctypes.c_void_p(trees.ctypes.data), ctypes.c_int(ntrees)
)

#
# display the results
#

if outpath is None:
	orig = img
	pred = predictions.reshape(nrows, ncols, nchns)

	show = numpy.zeros((orig.shape[0], 2*orig.shape[1], 3), dtype=numpy.float32)

	show[:, 0:orig.shape[1], :] = orig
	show[:, orig.shape[1]:, :]  = pred

	cv2.imshow('original | approximated', show)
	cv2.waitKey(0)
else:
	cv2.imwrite(outpath, predictions.reshape(nrows, ncols, nchns))
