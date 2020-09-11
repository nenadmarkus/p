#
# released under the MIT license
# https://tehnokv.com/posts/visualizing-audio-with-cppns
#

import numpy
import sys
import os
import time

#
if len(sys.argv)!=3:
	print('* args: <input:audio> <output:video>')
	sys.exit()

audiopath = sys.argv[1]
vidpath = sys.argv[2]

#
seed = None
show = False

start = time.time()

try:
	import torchaudio
	sound, fs = torchaudio.load(audiopath)
	sound = sound.numpy()[:, 0]
except ImportError:
	from audio_loader import load_audio
	sound, fs = load_audio(audiopath)

print('* elapsed time (feature extraction): %d [ms]' % int(1000*(time.time() - start)))

if fs!=44100:
	print('* fs = %d [kHz]' % fs)
	print('* sample rate should be 44.1 [kHZ] -> aborting ...')
	sys.exit()

#
#
#

def condense_spectrum(ampspectrum):
	'''
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
	'''
	'''
	#
	bands = numpy.zeros(4, dtype=numpy.float32)
	#
	bands[0] = numpy.sum(ampspectrum[0:4])
	bands[1] = numpy.sum(ampspectrum[4:12])
	bands[2] = numpy.sum(ampspectrum[12:60])
	bands[3] = numpy.sum(ampspectrum[60:])
	'''
	#
	bands = numpy.zeros(5, dtype=numpy.float32)
	#
	bands[0] = numpy.sum(ampspectrum[0:4])
	bands[1] = numpy.sum(ampspectrum[4:12])
	bands[2] = numpy.sum(ampspectrum[12:28])
	bands[3] = numpy.sum(ampspectrum[28:124])
	bands[4] = numpy.sum(ampspectrum[124:])
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

start = time.time()

amps = do_stft(sound, fs, fps)
amps = 0.5*amps/numpy.median(amps, 0)
amps[amps < 0.1] = 0.0

print('* elapsed time (feature extraction): %d [ms]' % int(1000*(time.time() - start)))

#
#
#

import cv2

nrows = 512
ncols = 512

rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(nrows/2.0)
colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(ncols/2.0)

#
#
#

if seed is not None:
	numpy.random.seed(seed)

nlayers = 4
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

def gen(layers, feature, cost):
	#
	inputs = [rowmat, colmat, numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
	inputs.append(feature*numpy.ones(rowmat.shape))
	inputs.append(cost*numpy.ones(rowmat.shape))
	#
	coordmat = numpy.stack(inputs).transpose(1, 2, 0)
	coordmat = coordmat.reshape(-1, coordmat.shape[2])

	result = coordmat.astype(numpy.float32)

	for layer in layers:
		#
		#result = numpy.tanh(numpy.matmul(result, layer))
		result = numpy.clip(numpy.matmul(result, layer), -1.0, 1.0)

	result = (1.0 + result)/2.0
	result[:, 1] = 1.0

	#
	radial = numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2)).reshape(nrows*ncols)
	R = min(1.3*feature + 1e-6, 1.0)
	window = 1.0 - (radial/R)**2
	window[radial>=R] = 0
	window = numpy.stack([window, window, window]).transpose()
	result = result * window

	#
	return result, window

#
#
#

def blend(i1, a1, i2, a2):
	#
	i = (a1*i1 + a2*i2)/(1e-6 + a1 + a2)
	a = numpy.max(numpy.stack((a1, a2)), 0)
	#
	return i, a

#
#
#

cppns = [build_cpnn() for i in range(0, 8)]

os.system('mkdir -p frames/')

n = amps.shape[0]
features = amps[0, :]

start = time.time()

for t in range(0, n):
	print('* %d/%d' % (t+1, n))
	#
	features = 0.8*features + 0.2*amps[t, :]

	#
	rends = []
	for j in range(0, len(features)):
		#
		if t==0 and False:
			result = gen(cppns[j], 1.0, numpy.cos(2*numpy.pi*t/(8*fps)))
		else:
			result, alpha = gen(cppns[j], features[j], numpy.cos(2*numpy.pi*t/(4*fps)))
		result = result.reshape(nrows, ncols, -1)
		result = 1.0 - result
		#
		alpha = alpha.reshape(nrows, ncols, 3)
		#
		rends.append((result, alpha))

	#
	#result = numpy.concatenate(rends, 1)
	#
	result = numpy.ones((3*nrows//2, 3*ncols, 3), dtype=numpy.float32)
	alpha = numpy.zeros((3*nrows//2, 3*ncols, 3), dtype=numpy.float32)
	#
	for i in range(0, len(features)):
		#
		if i%2==0:
			r1, r2 = 0, nrows
		else:
			r1, r2 = nrows//2, 3*nrows//2
		#
		c1 = i*ncols//2
		c2 = c1 + ncols
		#
		result[r1:r2, c1:c2, :], alpha[r1:r2, c1:c2, :] = blend(rends[i][0], rends[i][1], result[r1:r2, c1:c2, :], alpha[r1:r2, c1:c2, :])

	#
	result[alpha < 0.05] = 1.0

	#
	result = (255.0*result).astype(numpy.uint8)

	if show:
		cv2.imshow('...', result)
		cv2.waitKey(1)

	cv2.imwrite('frames/%06d.png' % t, result)

print('* elapsed time (rendering): %d [s]' % int(time.time() - start))

if show:
	cv2.destroyAllWindows()

#
#
#

os.system('rm %s' % vidpath)
os.system('ffmpeg -r ' + str(fps) + ' -f image2 -s 64x64 -i frames/%06d.png -i ' + audiopath + ' -crf 25 -vcodec libx264 -pix_fmt yuv420p ' + vidpath)
os.system('rm -rf frames/')