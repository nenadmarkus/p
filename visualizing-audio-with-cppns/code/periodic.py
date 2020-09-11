import numpy
import sys
import os
import time

#
seed = None
show = False

#
#
#

import cv2

nrows = 512
ncols = 1024

rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(nrows/2.0)
colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(ncols/2.0)

#
#
#

if seed is not None:
	numpy.random.seed(seed)

nlayers = 6
hsize = 16

def build_cpnn():
	#
	cppn = []

	for i in range(0, nlayers):
		#
		if i == 0:
			mutator = numpy.random.randn(3 + 2, hsize)
		elif i==nlayers-1:
			mutator = numpy.random.randn(hsize, 3)
		else:
			mutator = numpy.random.randn(hsize, hsize)

		#
		mutator = mutator.astype(numpy.float32)

		#
		cppn.append(mutator)
	#
	return cppn

def gen_frame(layers, cost, sint):
	#
	inputs = [rowmat, colmat, numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
	inputs.append(cost*numpy.ones(rowmat.shape))
	inputs.append(sint*numpy.ones(rowmat.shape))
	#
	coordmat = numpy.stack(inputs).transpose(1, 2, 0)
	coordmat = coordmat.reshape(-1, coordmat.shape[2])

	result = coordmat.astype(numpy.float32)

	for layer in layers:
		#
		result = numpy.tanh(numpy.matmul(result, layer))
		#result = numpy.clip(numpy.matmul(result, layer), -1.0, 1.0)

	result = (1.0 + result)/2.0
	#result[:, 1] = 1.0

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

cppn = build_cpnn()

os.system('mkdir -p frames/')

n = 360
freq = 1.0/float(n)

for t in range(0, n):
	print('* %d/%d' % (t+1, n))
	#
	result = result = gen_frame(cppn, numpy.cos(2*numpy.pi*freq*t), numpy.sin(2*numpy.pi*freq*t))
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