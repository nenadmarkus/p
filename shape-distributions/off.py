#
import os
import sys
import ctypes
import numpy

# "Efficient and flexible sampling with blue noise properties of triangular meshes"
# https://stackoverflow.com/questions/27224934/can-i-generate-point-cloud-from-mesh
# 

#
if len(sys.argv)!=3:
	print('* args: <off-model> <output-file-path>')
	sys.exit()

#
# compile the lib
#

os.system('cc off.c -fPIC -shared -o lib.so')
lib = ctypes.cdll.LoadLibrary('./lib.so')
os.system('rm lib.so')

#
# load the model in OFF format
#

ls = open(sys.argv[1], 'r').read().split()

#
nverts = int(ls[1])
nfaces = int(ls[2])

n = 4
verts = []
for i in range(0, nverts):
	verts.append(
		[float(ls[n+0]), float(ls[n+1]), float(ls[n+2])]
	)
	n += 3
verts = numpy.array(verts, dtype=numpy.float32)

faces = []
for i in range(0, nfaces):
	#
	if int(ls[n+0]) != 3:
		print('* error: only triangles supported')
	#
	faces.append(
		[int(ls[n+1]), int(ls[n+2]), int(ls[n+3])]
	)
	n += 4
faces = numpy.array(faces, dtype=numpy.int32)

#
#
#

nsamples = 1024
samples = numpy.zeros((nsamples, 3), dtype=numpy.float32)

lib.sample_surface_points(
	ctypes.c_void_p(samples.ctypes.data), ctypes.c_int(samples.shape[0]),
	ctypes.c_void_p(verts.ctypes.data), ctypes.c_int(verts.shape[0]),
	ctypes.c_void_p(faces.ctypes.data), ctypes.c_int(faces.shape[0])
)