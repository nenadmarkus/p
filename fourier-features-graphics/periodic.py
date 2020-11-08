import numpy
import sys
import os
import time
import cv2

seed = None
show = True

A = 1.0
L = 6
O = 3
H = 16
F = 32

nrows = 512
ncols = 1024

#
#
#

if seed is not None:
	numpy.random.seed(seed)

def build_cpnn():
	#
	freqs = A*numpy.random.randn(F, 3).astype(numpy.float32)
	freqs[:, 2] = 1.0
	#
	weights = []
	for i in range(0, L):
		if i == 0:
			w = numpy.random.randn(2*F, H)
		elif i==L-1:
			w = numpy.random.randn(H, O)
		else:
			w = numpy.random.randn(H, H)
		w = w.astype(numpy.float32)
		weights.append(w)
	#
	return freqs, weights
#
#
#

# construct a 2D array in which each row has integers between 0 and nrows-1
rowmat = numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T
# construct a 2D array in which each column has integers between 0 and ncols-1
colmat = numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)

# normalize coordinates
rowmat = rowmat/max(nrows, ncols) - 0.5
colmat = colmat/max(nrows, ncols) - 0.5

def gen_frame(freqs, weights, t):
	#
	inputs = [rowmat, colmat, t*numpy.ones(rowmat.shape)]
	#
	inputs = numpy.stack(inputs).transpose(1, 2, 0)
	inputs = inputs.reshape(-1, inputs.shape[2]).astype(numpy.float32)

	#
	inputs = 2*numpy.pi*numpy.matmul(inputs, freqs.T)
	inputs = numpy.concatenate([numpy.cos(inputs), numpy.sin(inputs)], 1)

	#
	result = inputs

	for w in weights:
		#
		result = numpy.tanh(numpy.matmul(result, w))
		#result = numpy.clip(numpy.matmul(result, layer), -1.0, 1.0)

	result = (1.0 + result)/2.0

	#
	'''
	radial = numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2)).reshape(nrows*ncols)
	R = 1.0
	window = 1.0 - (radial/R)**2
	window[radial>=R] = 0
	window = numpy.stack([window, window, window]).transpose()
	result = result * window
	'''

	#
	return result

#
#
#

freqs, weights = build_cpnn()

os.system('mkdir -p frames/')

n = 360
freq = 1.0/float(n)

for t in range(0, n):
	print('* %d/%d' % (t+1, n))
	#
	result = gen_frame(freqs, weights, t/n)
	result = result.reshape(nrows, ncols, -1)
	#result = 1.0 - result
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
