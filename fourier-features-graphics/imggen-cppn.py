import sys
import numpy

if len(sys.argv) == 1:
	A = 1.0
	P = None
elif len(sys.argv) == 2:
	A = float(sys.argv[1])
	P = None
elif len(sys.argv) == 3:
	A = float(sys.argv[1])
	P = sys.argv[2]
else:
	print("* args: [stdev] [output path]")
	sys.exit()
#
#
#

L = 4
O = 3
H = 16
nrows = 512
ncols = 512

# construct a 2D array in which each row has integers between 0 and nrows-1
rowmat = numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T
# construct a 2D array in which each column has integers between 0 and ncols-1
colmat = numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)

# normalize coordinates
rowmat = rowmat/max(nrows, ncols) - 0.5
colmat = colmat/max(nrows, ncols) - 0.5

# compute random Fourier features
inputs = []
for i in range(0, 32):
	#
	b1 = A*numpy.random.randn()
	b2 = A*numpy.random.randn()
	#
	inputs.append(numpy.cos(2*numpy.pi*(b1*rowmat + b2*colmat)))
	inputs.append(numpy.sin(2*numpy.pi*(b1*rowmat + b2*colmat)))

inputs = numpy.stack(inputs).transpose(1, 2, 0).reshape(-1, len(inputs))
inputs = numpy.ascontiguousarray(inputs.astype(numpy.float32))

#
#
#

results = inputs.copy()
for i in range(0, L):
	if i==L-1:
		W = numpy.random.randn(results.shape[1], O)
	else:
		W = numpy.random.randn(results.shape[1], H)
	results = numpy.tanh(numpy.matmul(results, W))
# rescale the input to (0.0, 1.0)
results = (1 + results)/2.0
# reshape the result into an image and convert its pixels to uint8 numbers
results = (255.0*results.reshape(nrows, ncols, results.shape[-1])).astype(numpy.uint8)
# optional: save the result to file using OpenCV
import cv2
if P is None:
	cv2.imshow("gen", results)
	cv2.waitKey(0)
else:
	cv2.imwrite(P, results)
