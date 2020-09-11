import numpy
from glumpy import app, gl, glm, gloo
import os
import cv2

NSTACKS = 512
NSLICES = 512
WSIZE = 512
NFRAMES = 360
DRAW_OUTLINE = False

os.system('mkdir -p frames/')
nframes = 0
def store_frame():
    #
    global nframes
    #
    img = numpy.frombuffer(gl.glReadPixels(0, 0, WSIZE, WSIZE, gl.GL_RGB, gl.GL_UNSIGNED_BYTE), dtype=numpy.uint8)
    img = img.reshape(WSIZE, WSIZE, 3)
    #cv2.putText(img, '@tehnokv', (6*WSIZE//10, 9*WSIZE//10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('frames/%06d.png' % nframes, img)
    nframes = nframes + 1

vertex = """
uniform mat4   model;         // Model matrix
uniform mat4   view;          // View matrix
uniform mat4   projection;    // Projection matrix
uniform vec4   u_color;       // Global color
attribute vec4 color;         // Vertex color
attribute vec3 position;      // Vertex position
varying vec4   v_color;       // Interpolated fragment color (out)
void main()
{
    v_color = u_color*color;
    gl_Position = projection * view * model * vec4(position,1.0);
}
"""

fragment = """
varying vec4 v_color; // Interpolated fragment color (in)
void main()
{
    gl_FragColor = v_color;
}
"""

window = app.Window(width=WSIZE, height=WSIZE, color=(1, 1, 1, 1))

@window.event
def on_draw(dt):
    global phi, theta
    window.clear()

    if not DRAW_OUTLINE:
        # fill
        cube['u_color'] = 1, 1, 1, 1
        cube.draw(gl.GL_TRIANGLES, I)
    else:
        # outline
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glEnable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_FALSE)
        cube['u_color'] = 0, 0, 0, 1
        cube.draw(gl.GL_LINES, L)
        gl.glDepthMask(gl.GL_TRUE)

    # rotate
    print(phi)
    phi += 360.0/NFRAMES # degrees
    model = numpy.eye(4, dtype=numpy.float32)
    glm.rotate(model, phi, 0, 1, 0)
    cube['model'] = model

    store_frame()

@window.event
def on_resize(width, height):
    cube['projection'] = glm.perspective(45.0, width / float(height), 2.0, 100.0)

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glCullFace(gl.GL_FRONT)
    gl.glEnable(gl.GL_CULL_FACE)

#
#
#

def build_cube():
    #
    vertices = numpy.array([[ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],
                       [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], [-1,-1,-1]], dtype=numpy.float32)
    colors = numpy.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                       [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]], dtype=numpy.float32)
    indices =  numpy.array([0,1,2, 0,2,3,  0,3,4, 0,4,5,  0,5,6, 0,6,1,
                  1,6,7, 1,7,2,  7,4,3, 7,3,2,  4,7,6, 4,6,5], dtype=numpy.uint32)
    #
    return vertices, colors, indices
#
#
#

def gen_cppn(inp_size, otp_size, hsize=32, nlayers=8):
    #
    layers = []
    for i in range(0, nlayers):
        #
        if i == 0:
            mutator = numpy.random.randn(inp_size, hsize)
        elif i==nlayers-1:
            mutator = numpy.random.randn(hsize, otp_size)
        else:
            mutator = numpy.random.randn(hsize, hsize)
        #
        mutator = mutator.astype(numpy.float32)

        #
        layers.append(mutator)
    #
    return layers

def run_cppn(coordmat, noutputs, hsize=24, nlayers=8):
    #
    layers = gen_cppn(coordmat.shape[1], noutputs, hsize=hsize, nlayers=nlayers)
    #
    result = coordmat.copy().astype(numpy.float32)

    for layer in layers:
        #
        result = numpy.clip(numpy.matmul(result, layer), -1.0, 1.0)

    result = (1.0 + result)/2.0

    result = result

    return result

# the following funciton could be heavily optimized
# (not a priority at this point)
def build_sphere(radius=1.0):
    #
    nstacks = NSTACKS
    nslices = NSLICES
    #
    indices = []
    lines = []
    angles = []

    for stack in range(0, nstacks):
        #
        theta1 = (stack+0)/float(nstacks)*numpy.pi
        theta2 = (stack+1)/float(nstacks)*numpy.pi
        #
        for slice in range(0, nslices):
            #
            phi1 = (slice+0)/float(nslices)*2*numpy.pi
            phi2 = (slice+1)/float(nslices)*2*numpy.pi
            #
            angles.extend([[theta1, phi1], [theta1, phi2], [theta2, phi1], [theta2, phi2]])
            #
            i11 = len(angles) + 0
            i12 = len(angles) + 1
            i21 = len(angles) + 2
            i22 = len(angles) + 3
            #
            if stack==0:
                # north pole
                indices.extend([i11, i22, i21])
                lines.extend([i11, i22, i11, i21, i22, i21])
            elif stack+1==nstacks:
                # south pole
                indices.extend([i22, i11, i12])
                lines.extend([i22, i11, i22, i12, i11, i12])
            else:
                # body
                indices.extend([i11, i12, i21])
                lines.extend([i11, i22, i11, i21, i22, i21])
                indices.extend([i12, i22, i21])
                lines.extend([i22, i11, i22, i12, i11, i12])
    #
    angles = numpy.array(angles, dtype=numpy.float32)
    #
    vertices = numpy.stack([
        numpy.sin(angles[:, 0])*numpy.cos(angles[:, 1]),
        numpy.sin(angles[:, 0])*numpy.sin(angles[:, 1]),
        numpy.cos(angles[:, 0])
    ]).transpose()
    #
    colors = numpy.ones((len(vertices), 4), dtype=numpy.float32)
    colors[:, 0:3] = run_cppn(vertices, 3)
    #
    rmod = 0.0*(0.5-run_cppn(vertices, 1, hsize=16, nlayers=8)).reshape(-1, 1)
    vertices = vertices + rmod*vertices
    #
    return vertices, colors, numpy.array(indices, dtype=numpy.uint32), numpy.array(lines, dtype=numpy.uint32)

#
#
#

vertices, colors, indices, lines = build_sphere()

V = numpy.zeros(len(vertices), [("position", numpy.float32, 3), ("color", numpy.float32, 4)])
V["position"] = vertices
V["color"]    = colors
V = V.view(gloo.VertexBuffer)
I = indices
I = I.view(gloo.IndexBuffer)
L = lines
L = L.view(gloo.IndexBuffer)

cube = gloo.Program(vertex, fragment)
cube.bind(V)

cube['model'] = numpy.eye(4, dtype=numpy.float32)
cube['view'] = glm.translation(0, 0, -5)
phi, theta = 0.0, 0.0

app.run(framerate=60, framecount=NFRAMES)

os.system('ffmpeg -y -r 60 -i frames/%06d.png -crf 25 -vcodec libx264 -pix_fmt yuv420p out.mp4')
os.system('rm -rf frames/')