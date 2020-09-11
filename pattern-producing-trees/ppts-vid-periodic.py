import numpy
import sys
import os
import time
import cv2
import ctypes

show = True

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

def build_trees(ntrees, tdepth):
	inputs = numpy.zeros((16384, 4))
	inputs[:, 0:2] = 2.0*numpy.random.rand(inputs.shape[0], 2) - 1.0
	wt = 2*numpy.pi*numpy.random.rand(inputs.shape[0])
	inputs[:, 2] = numpy.cos(wt)
	inputs[:, 3] = numpy.sin(wt)

	targets = numpy.ones((inputs.shape[0], 3))
	targets[:, 0:2] = inputs[:, 0:2]
	targets[:, 2] = inputs[:, 2]*inputs[:, 3]

	inputs = inputs.astype(numpy.float32)
	targets = targets.astype(numpy.float32)

	trees = numpy.zeros(ntrees, dtype=numpy.int64)
	ud3lib.new_ensemble(
		ctypes.c_float(0.25),
		ctypes.c_void_p(trees.ctypes.data), ctypes.c_int(ntrees), ctypes.c_int(tdepth),
		ctypes.c_void_p(targets.ctypes.data), ctypes.c_int(targets.shape[1]),
		ctypes.c_void_p(inputs.ctypes.data), ctypes.c_int(inputs.shape[1]),
		ctypes.c_void_p(0),
		ctypes.c_int(inputs.shape[0]),
		ctypes.c_int(32)
	)

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

	return trees

#
#
#

nrows = 512
ncols = 512

rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(min(nrows, ncols)/2.0)
colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(min(nrows, ncols)/2.0)

def gen_frame(trees, cost, sint):
	#
	inp = [rowmat, colmat, cost*numpy.ones(rowmat.shape), sint*numpy.ones(rowmat.shape)]
	inp = numpy.stack(inp).transpose(1, 2, 0).reshape(-1, len(inp))
	inp = numpy.ascontiguousarray(inp.astype(numpy.float32))

	results = numpy.zeros((inp.shape[0], 3), dtype=numpy.float32)

	ud3lib.run_ensemble(
		ctypes.c_void_p(trees.ctypes.data), ctypes.c_int(len(trees)),
		ctypes.c_void_p(inp.ctypes.data), ctypes.c_int(inp.shape[1]),
		ctypes.c_void_p(results.ctypes.data), ctypes.c_int(results.shape[1]),
		ctypes.c_int(inp.shape[0])
	)

	return results

#
#
#

start = time.time()
trees = build_trees(16, 12)
print('* trees built in %d [s]' % int(time.time() - start))

os.system('mkdir -p frames/')

n = 360
freq = 1.0/float(n)

for t in range(0, n):
	result = gen_frame(trees, numpy.cos(2*numpy.pi*freq*t), numpy.sin(2*numpy.pi*freq*t))
	result = result.reshape(nrows, ncols, -1)
	result = (255.0*result).astype(numpy.uint8)

	if show:
		cv2.imshow('...', result)
		cv2.waitKey(1)

	cv2.imwrite('frames/%06d.png' % t, result)

if show:
	cv2.destroyAllWindows()

#
#
#

os.system('rm out.mp4')
os.system('ffmpeg -r 60 -f image2 -s 64x64 -i frames/%06d.png -crf 25 -vcodec libx264 -pix_fmt yuv420p out.mp4')
os.system('rm -rf frames/')