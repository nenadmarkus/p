# 
# modified from <https://gist.github.com/kylemcdonald/85d70bf53e207bab3775>
#

import numpy
import subprocess
import os
DEVNULL = open(os.devnull, 'w')

# load_audio can not detect the input type
def load_audio(filename, normalize=True):
    #
    nchannels = 1
    fs = 44100
    #
    cmd = ['ffmpeg', '-i', filename, '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(fs), '-ac', '1', '-']
    #
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=DEVNULL, bufsize=4096)
    bytes_per_sample = 4
    chunk_size = nchannels * bytes_per_sample * fs # read in 1-second chunks
    #
    raw = b''
    stop = False
    while not stop:
        data = p.stdout.read(chunk_size)
        if data:
            raw += data
        else:
            stop = True
    #
    audio = numpy.fromstring(raw, dtype=numpy.float32)
    if audio.size == 0:
        return audio, fs
    #
    if nchannels > 1:
        audio = audio.reshape((-1, nchannels)).transpose()
    #
    if normalize:
        peak = numpy.abs(audio).max()
        if peak > 0:
            audio /= peak
    #
    return audio, fs