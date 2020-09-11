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

try:
	import torchaudio
	sound, fs = torchaudio.load(audiopath)
	sound = sound.numpy()[:, 0]
except ImportError:
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

import cv2

nrows = 64
ncols = 64

rowmat = (numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T - nrows/2.0)/(nrows/2.0)
colmat = (numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)   - ncols/2.0)/(ncols/2.0)

#
#
#

window = 1.0 - numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2)).reshape(nrows*ncols)
window[window<0] = 0.0
window = numpy.stack([window, window, window]).transpose()

if seed is not None:
	numpy.random.seed(seed)
nlayers = 8
hsize = 16
layers = []

for i in range(0, nlayers):
	#
	if i == 0:
		mutator = numpy.random.randn(3 + amps.shape[1], hsize)
	elif i==nlayers-1:
		mutator = numpy.random.randn(hsize, 3)
	else:
		mutator = numpy.random.randn(hsize, hsize)
	#
	mutator = mutator.astype(numpy.float32)

	#
	layers.append(mutator)

def gen(features):
	#
	fmaps = [f*numpy.ones(rowmat.shape) for f in features]
	#
	inputs = [rowmat, colmat, numpy.sqrt(numpy.power(rowmat, 2)+numpy.power(colmat, 2))]
	inputs.extend(fmaps)
	#
	coordmat = numpy.stack(inputs).transpose(1, 2, 0)
	coordmat = coordmat.reshape(-1, coordmat.shape[2])

	result = coordmat.copy().astype(numpy.float32)

	for layer in layers:
		#
		result = numpy.tanh(numpy.matmul(result, layer))

	result = (1.0 + result)/2.0
	#result[:, 0] = 0

	result = result * window

	return result

#
#
#

os.system('mkdir -p frames/')

n = amps.shape[0]
features = amps[0, :]

start = time.time()

for t in range(0, n):
	print('* %d/%d' % (t+1, n))
	#
	features = 0.9*features + 0.1*amps[t, :]
	#
	result = gen( features )
	result = (255.0*result.reshape(nrows, ncols, -1)).astype(numpy.uint8)
	#
	#result = 255 - result
	#result = cv2.resize(result, (256, 256))
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
