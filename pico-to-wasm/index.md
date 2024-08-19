---
title: Compiling a face detector written in C to WebAssembly
date: 2018-04-15 (updated on 2020-09-12)
excerpt: How to use Emscripten to compile https://github.com/nenadmarkus/pico to WebAssembly and run it inside a web browser.
tags: pico, face detection, webassembly, emscripten, object detection, viola-jones, decision tree, boosting, non-maximum suppression
layout: default
katex: false
---

In our [last post](../picojs-intro), we described [`pico.js`](https://github.com/nenadmarkus/picojs), a library for [real-time face detection](../picojs-intro/demo) written in 200 lines of JavaScript.
The original implementation of [`pico`](https://arxiv.org/pdf/1305.4537.pdf) is written in C: <https://github.com/nenadmarkus/pico>.
Here we show how to compile its runtime part to WebAssembly.

Highlights of this post:

* a real-time, in-browser face detector (see [webcam demo](demo/));
* a step-by-step guide explaining how to compile C code to Wasm.

## About WebAssembly

The [official page](http://webassembly.org/) of the WebAssembly project contains the following definition:

> WebAssembly (abbreviated Wasm) is a binary instruction format for a stack-based virtual machine. Wasm is designed as a portable target for compilation of high-level languages like C/C++/Rust, enabling deployment on the web for client and server applications.

The [wikipedia article](https://en.wikipedia.org/wiki/WebAssembly) mentions that Wasm should complement JavaScript to speed up performance-critical parts of web applications and later on to enable web development in other languages than JavaScript.

The main advantages that Wasm aims to deliver are

* load-time efficiency;
* smaller memory footprint;
* improved execution speed and related benefits (e.g., prolonged battery life for mobile devices).

This sounds doable and very promising.
After all, Wasm is a binary format that is intended to be compiled to machine code.

From recently (end of 2017), all major browsers support Wasm (see [this](https://blog.mozilla.org/blog/2017/11/13/webassembly-in-browsers/) blog post).

Let us now go through how to port the official (C-based) [`pico`](https://github.com/nenadmarkus/pico) to Wasm.

## Porting `pico` to Wasm

The following subsections talk about:

* Emscripten and how to set up its C compiler, `emcc`;
* C glue around `pico` and how to compile it to Wasm;
* instantiating and using the obtained Wasm.

All are more-or-less self-contained, so you can skip some parts if they introduce something you already know.

### The Emscripten toolchain

We will use the [Emscripten toolchain](https://en.wikipedia.org/wiki/Emscripten) for compiling C to Wasm.
The [official website](http://kripken.github.io/emscripten-site/) introduces it as follows:

> Emscripten is a toolchain for compiling to asm.js and WebAssembly, built using LLVM, that lets you run C and C++ on the web at near-native speed without plugins.

It is an impressive piece of software that enabled enthusiasts to [port](https://github.com/kripken/emscripten/wiki/Porting-Examples-and-Demos) a lot of C/C++ apps to the web environment.

Our initial goal is to setup the Emscripten C compiler, `emcc`. 
This can be done by installing the Emscripten SDK: <https://developer.mozilla.org/en-US/docs/WebAssembly/C_to_wasm#Emscripten_Environment_Setup>.
A nice thing about this SDK is that it's portable:
everything is contained in a single folder and it does not mess with the configuration of your system.

The following sections assume that `emcc` is available on the system.

### Wrapping the official `pico` code and compiling it to Wasm

Let us start by downloading the runtime from the official repo:

	wget https://github.com/nenadmarkus/pico/raw/346881039e5d1f5abe64733a49886bdfd5ab2d51/rnt/picornt.c

This small C file contains a function `find_objects` that can be used to detect faces in images when invoked with proper parameters.
One of the parameters that needs to be passed to this function is the *detection cascade*.
The detection cascade can be seen as a small brain that is able to discern objects of interest from image background.
One such detection cascade designed to find faces in images is named `facefinder`.
Let us download it from the official repository:

	wget https://github.com/nenadmarkus/pico/raw/346881039e5d1f5abe64733a49886bdfd5ab2d51/rnt/cascades/facefinder

Note that its size is around 250kB and that this will roughly equal the size of the output Wasm file as we will plug it in directly.
However, before we can do that, we need to transform it into a C-compatible hexadecimal array with the following command:

	cat facefinder | hexdump -v -e '16/1 "0x%x," "\n"' > facefinder.hex

You can open `facefinder.hex` with a text editor and view its contents.

Next, let us make a simple wrapper `main.c` aroud the `pico` runtime:

```c
#include "picornt.c"

int find_faces(
	float rcsq[],
	int maxndetections,
	unsigned char pixels[],
	int nrows,
	int ncols,
	int ldim,
	float scalefactor,
	float shiftfactor,
	int minfacesize,
	int maxfacesize
)
{
	static char facefinder[] = {
		#include "facefinder.hex"
	};

	static int slot = 0;
	static const int nmemslots = 5;
	static const int maxslotsize = 1024;
	static float memory[4*nmemslots*maxslotsize];
	static int counts[nmemslots];

	int n = find_objects(
		rcsq, maxndetections,
		facefinder,
		0.0f,
		pixels, nrows, ncols, ldim,
		scalefactor, shiftfactor,
		minfacesize, maxfacesize
	);

	n = update_memory(
		&slot,
		memory, counts, nmemslots, maxslotsize,
		rcsq, n, maxslotsize
	);

	n = cluster_detections(rcsq, n);

	return n;
}
```

Notice that `facefinder.hex` will be included into this C program via a preprocessor `#include` directive.

The reader of this post might be confused by a large number of parameters needed by the function `find_objects`.
Please go and take a look at the [official sample](https://github.com/nenadmarkus/pico/blob/346881039e5d1f5abe64733a49886bdfd5ab2d51/rnt/sample/sample.c) for more details.
For our purposes, it suffices to say that `rcsq` is an array of 4 times `maxndetections` that will hold the detection results after `pico` finishes processing the image.
We can set `maxndetections` to some reasonable number, e.g., 1024 as we do not expect more faces than that.
The array `pixels` holds the grayscale pixel values of the image.
Both `rcsq` and `pixels` need to be allocated in advance by the user.
Parameters `minfacesize` and `maxfacesize` are self-explanatory and should be set to, e.g., 100 and 1000 for real-time performance, respectively.
A good value for `scalefactor` is 1.1 and `shiftfactor` can be set to 0.1.
Thus, if we assume that the image is of size 480x640, we can invoke `find_faces` as follows:

	int nfaces = find_faces(rcsq, 1024, pixels, 480, 640, 640, 1.1f, 0.1f, 100, 1000);

The variable `nfaces` now contains the number of faces found in the image and `rcsq` is filled with their positions, sizes and detection quality.

The idea is to expose the `find_faces` function to JavaScript and invoke it from there.
We will do this through WebAssembly.
The Emscripten C compiler will help us with this:

	emcc main.c -o wasmpico.js -O3 -s EXPORTED_FUNCTIONS="['_find_faces', '_cluster_detections', '_malloc', '_free']" -s WASM=1

This will generate two files: `wasmpico.js` and `wasmpico.wasm`.
The file `wasmpico.js` contains the boilerplate code to load `wasmpico.wasm`.
Since we have passed the `-s EXPORTED_FUNCTIONS="['_find_faces', '_malloc', '_free']"` flag to `emcc`, the following functions will be available to JavaScript:

* `find_faces` (usage as explained above);
* `malloc` and `free` (enable us to allocate memory for `pixels` and `rcsq`).

A build script encapsulating the explained steps is available [here](demo/build.sh) (also: [main.c](demo/main.c)).

Let us now see how to use `wasmpico` from JavaScript.

### Running `wasmpico` in your JavaScript program

First, include `wasmpico.js` code:

```
<script src="wasmpico.js"></script>
```

The whole thing loads asynchronously, so you cannot use its functionality instantly.
We assume in the following text that this initialization process has finished and the object `Module` is available for use.

Let us first allocate memory for the operation of our Wasm module:

```javascript
const nrows=480, ncols=640;
const ppixels = Module._malloc(nrows*ncols);
const pixels = new Uint8Array(Module.HEAPU8.buffer, ppixels, nrows*ncols);

const maxndetections = 1024;
const prcsq = Module._malloc(4*4*maxndetections)
const rcsq = new Float32Array(Module.HEAPU8.buffer, prcsq, maxndetections);
```

We draw the image onto the canvas to retrieve its RGBA pixel values:

```javascript
// we assume these are the height and width of our image
const nrows=480, ncols=640;

const ctx = document.getElementsByTagName('canvas')[0].getContext('2d');
ctx.drawImage(image, 0, 0);
const rgba = ctx.getImageData(0, 0, ncols, nrows).data;
```

Next, we need to move this data into the memory of our Wasm module.
This, along with the conversion from RGBA to grayscale, can be done as follows:

```javascript
function rgba_to_grayscale(rgba, nrows, ncols) {
	for(let r=0; r<nrows; ++r)
		for(let c=0; c<ncols; ++c)
			// take just the green channel
			pixels[r*ncols + c] = rgba[r*4*ncols+4*c+1];
	return pixels;
}
```

Finally, we can now invoke the face detector and draw the found faces:

```javascript
// run the detector across the image
var ndetections = Module._find_faces(prcsq, maxndetections, ppixels, nrows, ncols, ncols, 1.1, 0.1, 100, 1000);

// draw detections
for(i=0; i<ndetections; ++i)
	// check the detection score
	// if it's above the (empirical) threshold, draw it
	if(rcsq[4*i+3]>50.0)
	{
		ctx.beginPath();
		ctx.arc(rcsq[4*i+1], rcsq[4*i+0], rcsq[4*i+2]/2, 0, 2*Math.PI, false);
		ctx.lineWidth = 3;
		ctx.strokeStyle = 'red';
		ctx.stroke();
	}
```

Be sure to look at the source code of the [webcam-based demo](demo/) if something is not clear after this exposition.
The demo is pretty well commented.

## Final notes

The potential of WebAssembly is huge.
Some preliminary experiments show that `wasmpico` is two times faster than [`pico.js`](https://github.com/nenadmarkus/picojs).
Note that the performance gap might improve further in the future as the technology matures.
