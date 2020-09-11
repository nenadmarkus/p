#
# released under the MIT license
#

import numpy
import sys
import os
import time
import cv2
import ctypes

#
if len(sys.argv)!=3:
	print('* args: <input:audio> <output:video>')
	sys.exit()

audiopath = sys.argv[1]
vidpath = sys.argv[2]

#
from audio_loader import load_audio
sound, fs = load_audio(audiopath)

if fs!=44100:
	print('* fs = %d [kHz]' % fs)
	print('* sample rate should be 44.1 [kHZ] -> aborting ...')
	sys.exit()

#
#
#

def condense_spectrum(ampspectrum):
	#
	bands = numpy.zeros(8, dtype=numpy.float32)
	#
	bands[0] = numpy.sum(ampspectrum[0:4])
	bands[1] = numpy.sum(ampspectrum[4:12])
	bands[2] = numpy.sum(ampspectrum[12:28])
	bands[3] = numpy.sum(ampspectrum[28:60])
	bands[4] = numpy.sum(ampspectrum[60:124])
	bands[5] = numpy.sum(ampspectrum[124:252])
	bands[6] = numpy.sum(ampspectrum[252:508])
	bands[7] = numpy.sum(ampspectrum[508:])
	#
	return bands

def do_stft(sound, fs, fps):
	#
	nsamples = len(sound)
	wsize = 2048
	stride = int(fs/fps)

	#
	amplitudes = []

	stop = False
	start = 0

	while not stop:
		#
		end = start + wsize
		if end > nsamples:
			end = nsamples
		#
		chunk = sound[start:end]

		if len(chunk) < 2048:
			padsize = 2048 - len(chunk)
			chunk = numpy.pad(chunk, (0, padsize), 'constant', constant_values=0)
		#
		freqspectrum = numpy.fft.fft(chunk)[0:1024]
		amplitudes.append( condense_spectrum(numpy.abs(freqspectrum)) )
		#
		start = start + stride

		if start >= nsamples:
			stop = True

	#
	return numpy.stack(amplitudes).astype(numpy.float32)

#
#
#

fps = 30

amps = do_stft(sound, fs, fps)
amps = 0.5*amps/numpy.median(amps, 0)

amps[amps < 0.1] = 0.0

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
	n = 65536
	inputs = numpy.zeros((n, 2 + amps.shape[1]))
	inputs[:, 0:2] = 2.0*numpy.random.rand(inputs.shape[0], 2) - 1.0

	for i in range(0, n):
		j = numpy.random.randint(amps.shape[0])
		inputs[i, 2:] = amps[j, :]

	targets = numpy.matmul(inputs, numpy.random.randn(2+amps.shape[1], 3))

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

rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(nrows/2.0)
colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(ncols/2.0)

def gen_frame(trees, features):
	fmaps = [f*numpy.ones(rowmat.shape) for f in features]

	inp = [rowmat, colmat]
	inp.extend(fmaps)
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

trees = build_trees(16, 15)

os.system('mkdir -p frames/')

n = amps.shape[0]
features = amps[0, :]

start = time.time()

for t in range(0, n):
	print('* %d/%d' % (t+1, n))
	#
	features = 0.9*features + 0.1*amps[t, :]
	#
	result = gen_frame(trees, features)
	result = (255.0*result.reshape(nrows, ncols, -1)).astype(numpy.uint8)
	#
	cv2.imshow('...', result)
	cv2.imwrite('frames/%06d.png' % t, result)
	cv2.waitKey(1)

print('* elapsed time (rendering): %d [s]' % int(time.time() - start))

cv2.destroyAllWindows()

#
#
#

os.system('rm %s' % vidpath)
os.system('ffmpeg -r ' + str(fps) + ' -f image2 -s 64x64 -i frames/%06d.png -i ' + audiopath + ' -crf 25 -vcodec libx264 -pix_fmt yuv420p ' + vidpath)
os.system('rm -rf frames/')