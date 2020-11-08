---
title: Generative graphics with random Fourier features
date: 2020-04-11
excerpt: Generate interesting visual patterns with positional encoding based on random Fourier features.
tags: random fourier features, generative graphics, decision trees, interesting patterns
layout: default
katex: true
---

Tancik et al. [1] recently performed a series of interesting experiments showing that positional encoding with random Fourier features (RFFs) significantly improves learning of high-frequency details on various tasks in computer graphics and image processing
(e.g., image regression, shape representation, MRI reconstruction and inverse rendering).
Their results are nicely illustrated in [this figure](https://user-images.githubusercontent.com/3310961/84946597-cdf59800-b09d-11ea-8f0a-e8aaeee77829.png).

Given the strong properties of RFFs, let us investigate if these can be combined with [pattern-producing networks](../visualizing-audio-with-cppns) and [trees](../pattern-producing-trees) to generate interesting abstract graphics.

## Random Fourier features (RFFs)

RFFs were originally introduced by Rahimi and Recht [2] for kernel approximation.
For our purposes, we define these features as:

$$
	\text{RFF}(\mathbf{v})=
	\left(
		\cos(2\pi\mathbf{f}_1^T\mathbf{v}), \sin(2\pi\mathbf{f}_1^T\mathbf{v}),
		\cos(2\pi\mathbf{f}_2^T\mathbf{v}), \sin(2\pi\mathbf{f}_2^T\mathbf{v}),
		\ldots,
		\cos(2\pi\mathbf{f}_N^T\mathbf{v}), \sin(2\pi\mathbf{f}_N^T\mathbf{v})
	\right)^T
$$

where $$\mathbf{v}$$ is the input vector to be transformed and there are $$N$$ frequency vectors $$\mathbf{f}_1, \mathbf{f}_2, \ldots, \mathbf{f}_N$$.
Each of these frequency vectors is sampled from a normal distribution with a zero mean and a diagonal covariance matrix of the form $$\sigma^2\mathbf{I}$$.
The standard deviation $$\sigma$$ is the most interesting property of the encoding as it sets the level of detail that can be represented.

## Positional encoding and generative graphics

Recall that [CPPNs](../visualizing-audio-with-cppns) are nothing else than [multilayer perceptrons (MLPs)](https://en.wikipedia.org/wiki/Multilayer_perceptron) with random weights that map an $$(x, y)$$ position into an RGB color value for that position:

$$
	(R, G, B) = \text{MLP}(x, y)
$$

However, let us transform $$\mathbf{v}=(x, y)^T$$ before it goes into the MLP:

$$
	(R, G, B) = \text{MLP}( RFF(x, y) )
$$

The produced visual patterns are interesting and qualitatively different than the ones in our previous [CPPN post](../visualizing-audio-with-cppns):

<center>
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/512x1024.jpg" style="width: 80%; max-width: 768;" alt="CPPN+RFF graphics">
</center>

<center>
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/e1.jpg" style="width: 40%; max-width: 384;" alt="CPPN+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/e2.jpg" style="width: 40%; max-width: 384;" alt="CPPN+RFF graphics">
</center>

The script [imggen-cppn.py](imggen-cppn.py) enables you to produce similar patterns for varying standard deviation $$\sigma$$.

	python3 imggen-cppn.py 2.0 out.jpg

Here are four examples for $$\sigma=1, 2, 3, 4$$:

<center>
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/1.jpg" style="width: 20%;" alt="CPPN+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/2.jpg" style="width: 20%;" alt="CPPN+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/3.jpg" style="width: 20%;" alt="CPPN+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/4.jpg" style="width: 20%;" alt="CPPN+RFF graphics">
</center>

We can observe that the amount of color mixing and spatial details steadily increases with $$\sigma$$.

We can also add a temporal dimension to produce videos such as the following one:

<div style="text-align:center;">
<video style="width: 70%; height:35%; max-width: 512px; max-height: 256px;" autoplay loop="" muted="" playsinline="">
<source src="https://drone.nenadmarkus.com/data/blog-stuff/rff/vid.mp4" type="video/mp4">
</video>
</div>

## Pattern-producing trees

Instead of an MLP, we can use [randomized trees](../pattern-producing-trees).
The code for this experiment is available [here](imggen-ppts.py).
We can invoke it as follows:

	python3 imggen-ppts.py 2.0 out.jpg

Here are some examples for varying standard deviation:

<center>
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/t1.jpg" style="width: 20%;" alt="PTTS+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/t2.jpg" style="width: 20%;" alt="PTTS+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/t3.jpg" style="width: 20%;" alt="PTTS+RFF graphics">
<img src="https://drone.nenadmarkus.com/data/blog-stuff/rffs/t4.jpg" style="width: 20%;" alt="PTTS+RFF graphics">
</center>

## Resources

[1] Tancik et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. [https://arxiv.org/abs/2006.10739](https://arxiv.org/abs/2006.10739), 2020

[2] Ali Rahimi and Ben Recht. Random Features for Large-Scale Kernel Machines. NIPS, 2007 
