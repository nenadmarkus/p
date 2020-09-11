import subprocess
import numpy
import os
import ctypes
import cv2
import time

#
#
#

os.system('cc api.c -O3 -fPIC -shared -o ud3.lib.so')
ud3lib = ctypes.cdll.LoadLibrary('./ud3.lib.so')
os.system('rm ud3.lib.so')

src = '''#include <stdint.h>
#include <stdlib.h>
void randomize_leaves(void* trees[], int ntrees, float newpreds[], int nnew)
{
	int i, j, k;

	// randomize predictions
	for(i=0; i<ntrees; ++i)
	{
		void* tree = trees[i];
		//
		int pdim = ((int32_t*)tree)[0];
		int tdepth = ((int32_t*)tree)[1];

		int32_t* finds = (int32_t*)&((int32_t*)tree)[2];
		float* threshs = (float*)&((int32_t*)tree)[2 + (1<<tdepth)-1];
		float* preds = (float*)&((int32_t*)tree)[2 + (1<<tdepth)-1 + (1<<tdepth)-1];

		//
		for(j=0; j<(1<<tdepth); ++j)
		{
			int u = rand()%nnew;

			for(k=0; k<pdim; ++k)
				preds[j*pdim +k] = newpreds[u*nnew + k];
		}
	}
}'''
f = open('randomizer.c', 'w')
f.write(src)
f.close()
os.system('cc randomizer.c -fPIC -shared -o randomizer.lib.so')
os.system('rm randomizer.c')
randomizerlib = ctypes.cdll.LoadLibrary('./randomizer.lib.so')
os.system('rm randomizer.lib.so')

#
#
#

nrows = 512
ncols = 512
nchns = 3

#
#
#

# construct a 2D array in which each row has numbers between -1.0 and 1.0
rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(min(nrows, ncols)/2.0)
# construct a 2D array in which each column has numbers between -1.0 and 1.0
colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(min(nrows, ncols)/2.0)

#inputs = [rowmat, colmat]
#inputs = [rowmat, colmat, numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
#inputs = [numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
inputs = [numpy.cos(numpy.arctan2(rowmat, colmat)), numpy.sin(numpy.arctan2(rowmat, colmat)), numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
#inputs = [numpy.cos(numpy.arctan2(rowmat, colmat)), numpy.sin(numpy.arctan2(rowmat, colmat)), numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2)), rowmat, colmat]

inputs = numpy.stack(inputs).transpose(1, 2, 0).reshape(-1, len(inputs))
inputs = numpy.ascontiguousarray(inputs.astype(numpy.float32))

inputs = numpy.matmul(inputs, numpy.random.randn(inputs.shape[1], inputs.shape[1]).astype(numpy.float32))

targets = numpy.stack([rowmat, colmat, numpy.ones((nrows, ncols))]).transpose(1, 2, 0).reshape(-1, 3)
targets = numpy.ascontiguousarray(targets)

#
#
#

ntrees = 8
tdepth = 8

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

print('* elapsed time (learning): %d' % int(time.time() - start))

#
#
#

predictions = numpy.zeros(targets.shape, dtype=numpy.float32)

start = time.time()

newpreds = numpy.array([
	[1.0, 0.0, 0.0],
	[0.0, 1.0, 0.0],
	[0.0, 0.0, 1.0]
], dtype=numpy.float32)/ntrees

randomizerlib.randomize_leaves(
	ctypes.c_void_p(trees.ctypes.data),
	ctypes.c_int(ntrees),
	ctypes.c_void_p(newpreds.ctypes.data),
	ctypes.c_int(newpreds.shape[0]),
)

ud3lib.run_ensemble(
	ctypes.c_void_p(trees.ctypes.data), ctypes.c_int(ntrees),
	ctypes.c_void_p(inputs.ctypes.data), ctypes.c_int(inputs.shape[1]),
	ctypes.c_void_p(predictions.ctypes.data), ctypes.c_int(predictions.shape[1]),
	ctypes.c_int(inputs.shape[0])
)

print('* elapsed time (prediction): %d' % int(time.time() - start))

pimg = (255.0*predictions.reshape(nrows, ncols, nchns)).astype(numpy.uint8)

cv2.imshow('approximated', pimg)
cv2.imwrite('out.jpg', pimg)
cv2.waitKey(0)