---
title: All-pairs Euclidean distance
date: 2019-05-10
excerpt: How to compute all-pairs Euclidean distance with efficient libraries for manipulating numerical arrays (e.g., numpy and pytorch).
tags: pairwise distance, Euclidean distance
layout: default
katex: true
---

Given two arrays of $d$-dimensional vectors, $\mathbf{A}\in\mathbb{R}^{M\times d}$ and $\mathbf{B}\in\mathbb{R}^{N\times d}$, we are interested in efficient vectorized code for obtaining a matrix $\mathbf{D}\in\mathbb{R}^{M\times N}$ such that

$$
	(\mathbf{D})_{ij} = \vert\vert\mathbf{a}_i - \mathbf{b}_j\vert\vert_2
$$

where $(\mathbf{D})_{ij}$ is the $ij$th entry of $\mathbf{D}$, $\mathbf{a}_i$ is the $i$th row of $\mathbf{A}$ and $\mathbf{b}_j$ is the $j$th row of $\mathbf{B}$.
In other words, we want to compute the Euclidean distance between all vectors in $\mathbf{A}$ and all vectors in $\mathbf{B}$.
The following `numpy` code does exactly this:

```
def all_pairs_euclid_naive(A, B):
	#
	D = numpy.zeros((A.shape[0], B.shape[0]), dtype=numpy.float32)
	for i in range(0, D.shape[0]):
		for j in range(0, D.shape[1]):
			D[i, j] = numpy.linalg.norm(A[i, :] - B[j, :])
	#
	return D
```

Unfortunately, this code is really inefficient.
To rectify the issue, we need to write a vectorized version in which we avoid the explicit usage of loops.
To arrive at a solution, we first expand the formula for the Euclidean distance:

$$
	\vert\vert\mathbf{a}_i - \mathbf{b}_j\vert\vert_2=
	(\mathbf{a}_i - \mathbf{b}_j)^T(\mathbf{a}_i - \mathbf{b}_j)=
	\mathbf{a}_i^T\mathbf{a}_i - 2\mathbf{a}_i^T\mathbf{b}_j + \mathbf{b}_j^T\mathbf{b}_j
$$

This leads us to the following equation for $\mathbf{D}$:

$$
	\mathbf{D} = \sqrt{\mathbf{S}_A - 2\cdot\mathbf{A}\cdot\mathbf{B}^T + \mathbf{S}_B}
$$

where $\mathbf{S}_A\in\mathbb{R}^{M\times N}$ is such that $(\mathbf{S}_A)_\{ij}=\mathbf{a}_i^T\mathbf{a}_i$ and
$\mathbf{S}_B\in\mathbb{R}^{M\times N}$ is such that $(\mathbf{S}_B)_{ij}=\mathbf{b}_j^T\mathbf{b}_j$.
The square root is taken elementwise.

Each of the three above terms can be obtain with vectorized code.

The first term is obtained for each $i$ by squaring the entries of $\mathbf{A}$, summing along the second dimension after that and repeating the obtained result $N$ times to obtain an $M\times N$ matrix.
The second term can be computed with the standard matrix-matrix multiplication routine.
The third term is obtained in a simmilar manner to the first term.

Without further ado, here is the `numpy` code:

```
def all_pairs_euclid_numpy(A, B):
	#
	sqrA = numpy.broadcast_to(numpy.sum(numpy.power(A, 2), 1).reshape(A.shape[0], 1), (A.shape[0], B.shape[0]))
	sqrB = numpy.broadcast_to(numpy.sum(numpy.power(B, 2), 1).reshape(B.shape[0], 1), (B.shape[0], A.shape[0])).transpose()
	#
	return numpy.sqrt(
		sqrA - 2*numpy.matmul(A, B.transpose()) + sqrB
	)
```

And the following implementation uses Pytorch:

```
def all_pairs_euclid_torch(A, B):
	#
	sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
	sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
	#
	return torch.sqrt(
		sqrA - 2*torch.mm(A, B.t()) + sqrB
	)
```

Note that the above code can be executed on the GPU as well.

By setting $M=2048$, $N=2048$ and $d=2048$, we obtain the following timings for the modern 40-thread Intel CPU and the Nvidia 1080Ti GPU:
<pre>
| Implementation | Time [ms] |
| -------------- | --------- |
| naive          | 29500     |
| numpy          | 160       |
| pytorch        | 20        |
| pytorch (cuda) | 15        |</pre>

You can get the time measurements for your hardware by running the following [script](code.py).
