---
title: Processing a numpy array in native code
date: 2019-01-14
excerpt: We show how to send a numpy array to native code for processing.
tags: ctypes, numpy, C
layout: default
katex: true
---

In this post, we show how to process a `numpy` array with native code written in C.
A self-contained example is available [here](test.py).

If you prefer a more verbose explanation than going through Python code, please read on.

## Introduction and motivation

The Python programming language is nowadays a *de facto* standard for prototyping/research in data science, computer vision, machine learning, etc.
However, writing algorithms in pure Python is computationally inefficient as it is an [interpreted language](https://en.wikipedia.org/wiki/Interpreted_language).
Thus, most scientific programs written in it use libraries like [`numpy`](https://en.wikipedia.org/wiki/NumPy) to perform numeric computations.

While `numpy` is a great library, some algorithms are difficult (or impossible) to express with its built-in functionalities.
Thus, for awesome performance, it might be desirable to pass an array to native code for processing.
To achieve this, we use [ctypes](https://docs.python.org/3/library/ctypes.html), a foreign function library for Python.
This library provides C compatible data types and allows to wrap functions in DLLs or shared libraries in pure Python.

Let us first formulate a dummy problem through which we will illustrate all the important points.

## Our dummy problem

Let `A` be an array consisting of `n` 32-bit floating point numbers.
Our task is to find the mean value of this array, i.e.,

$$
	\mu=
	\frac{1}{n}\sum_{i=0}^{n-1} A_i
$$

Using pure Python, this can be computed as:

```python
sum=0
for i in range(0, n):
	sum += A[i]
mean = sum/n
```

Trivial, of course.
However, it is even simpler using `numpy`:

```python
mean = numpy.mean(A)
```

In the next section, we express this computation in native code.

## Native code for computing the mean of an array

The following C function will do the job:

```python
float compute_mean(float* A, int n)
{
	int i;
	float sum = 0.0f;
	for(i=0; i<n; ++i)
		sum += A[i];
	return sum/n;
}
```

If we assume that the above code is in a file `lib.c`, we can compile a shared library as follows:

	cc lib.c -fPIC -shared -o lib.so

The following section shows how to invoke this function from Python.

## Using `ctypes` to call functions from a native library

To load `lib.so` as a Python object, execute the following commands:

```python
import ctypes
lib = ctypes.cdll.LoadLibrary('./lib.so')
```

The `compute_mean` function expects a pointer to the array and the number of elements within this array as parameters.
The first parameter can be obtained with `ctypes.c_void_p(A.ctypes.data)` and the second as `ctypes.c_int(n)`.

Next, we indicate that we expect a float as a return value and call the desired function on our array:

```python
lib.compute_mean.restype = ctypes.c_float
mean = lib.compute_mean(ctypes.c_void_p(A.ctypes.data), ctypes.c_int(n))
```

The default `restype` is `c_int`, so we do not have to set this flag in the case when the native function returns a C `int`.

## Additional remarks

A self-contained code for this tutorial is available [here](test.py).

A possible pitfall of this approach is forgetting that a `numpy` array can be stored in a non-continuous block of memory
(e.g., if we perform a [slicing operation](https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm) on a 2D array).
We can examine whether an array is contiguous by checking its `C_CONTIGUOUS` flag and, if required, react accordingly:

```python
if not A.flags['C_CONTIGUOUS']:
	A = numpy.ascontiguousarray(A)
```

## Resources

* <https://docs.python.org/3/library/ctypes.html>
* <https://stackoverflow.com/questions/5862915/passing-numpy-arrays-to-a-c-function-for-input-and-output>
* <https://stackoverflow.com/questions/29947639/cheapest-way-to-get-a-numpy-array-into-c-contiguous-order>
