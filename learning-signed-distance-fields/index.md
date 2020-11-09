---
title: Some experiments in learning signed distance fields from data
date: 2020-11-09
excerpt: Least squares fitting on random Fourier features seems to work well in some useful scenarios.
tags: random fourier features, signed distance bounds, least squares
layout: default
katex: true
---

There has been considerable interest in learning-based methods for signed distance field modeling.
And this is lately especially true in the area of deep learning:
see section "Implicit representation" at [https://github.com/subeeshvasu/Awsome_Deep_Geometry_Learning](https://github.com/subeeshvasu/Awsome_Deep_Geometry_Learning).

Thus, it is warranted to investigate whether [gridhopping](../fast-algo-sdb-to-mesh) and [lambdaCAD](https://lambdacad.gitlab.io) are useful in this area.
E.g., for extracting polygonal models from such representations for debugging or rendering purposes.

Here is a selection of interesting papers on geometric deep learning:

1. Zhiqin Chen and Hao Zhang. Learning Implicit Fields for Generative Shape Modeling. CVPR, 2019 ([arXiv](https://arxiv.org/abs/1812.02822))
2. Park et al. DeepSDF: Learning Continuous Signed Distance Functionsfor Shape Representation. CVPR, 2019 ([arXiv](https://arxiv.org/abs/1901.05103))
3. Li et al. Supervised Fitting of Geometric Primitives to 3D Point Clouds. CVPR, 2019 ([arXiv](https://arxiv.org/abs/1811.08988))
4. Davies et al. Overfit Neural Networks as a Compact Shape Representation. [https://arxiv.org/abs/2009.09808](https://arxiv.org/abs/2009.09808), 2020

These papers illustrate some core ideas and applications within the area.
The first two study the learning of **generative** models for 3D shapes.
I.e., for applications that need to generate novel 3D shapes belonging to a certain class, such as aeroplanes of cars.
This could be useful in computer games and some types of viritual reality software, for example.
In [1], the authors propose to model a shape as an occupancy map ($$+1$$ outside shape, $$-1$$ inside) with a neural network classifier.
Since this representaion is smooth, it is capable of defining shapes as implicit sufraces.
However, the approach would probably not work well with sphere tracing because of issues ontlined in a [previous post](../lipschitz-continuity-and-sphere-tracing) about [Lipschitz continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity).
On the other hand, the authors of [2] propose to model a 3D shape $$S$$ with a neural network that is learned to estimate the signed distance field:

$$
	\text{NN}(x, y, z)\approx
	d_S(x, y, z)
$$

Of course, there are some caveats that simplify the process and enable learning of a generative model.
Please see the paper for these details.
Even though the network is explicitly learned to approximate the distance to the shape, there is no guarantee that sphere tracing will work as intended (especially when far away from the shape),
but we expect less problems than with occupancy maps.
The paper by Li et al. [3] contains a similar approach to that of [2] except that it is not concerned with generative modeling, but with shape compression.
I.e., each shape is assigned a separate tiny network that approximates its signed distance field.
The hope is that this tiny network requires less memory to store than an explicit list of polygons, effectively enabling a compressed representation.
Thus, this approach has the same potential to be compatible with `gridhopping` as [2].
The fourth paper mentioned earlier (Davies et al. [4]) uses a neural network that outputs a list of primitives and their parameters to approximate an input point cloud.
Since the basic primitives used (spheres, cones, etc.) have explicit and efficiently computable signed distance fields, this approach is compatible with `gridhopping`.

Given the interesting resluts presented in the mentioned papers,
we experiment with two learning algorithms for converting a 3D polygonal mesh into a signed distance field.
The first one is based on random Fourier features and the second one on a simple feedforward neural network.
However, let us first describe the shapes we use and how we prepare the training data.

## Training data

We use the following 3D shapes:

<center>
<img src="https://raw.githubusercontent.com/nenadmarkus/gridhopping/master/implementations/python/experiments/models/all.png" style="width: 96%; max-width: 768;" alt="Shapes used in our experiments">
</center>

Our goal is to transform these shapes into signed distance fields.

Given such a shape $$S$$, we generate a trainig set of the form

$$
	\left\{(\mathbf{x}_i, d_i)\right\}_{i=1}^N
$$

where $$\mathbf{x}\in\mathbb{R}^3$$ is a point in 3D space and $$d_i$$ is the Euclidean distance from $$\mathbf{x}$$ to the surface of $$S$$.
This is conceptually very simple, but there are some subtleties.

In our experiments, $$S$$ is represented as a triangle mesh.
