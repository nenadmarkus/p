---
title: Fusing batch normalization and convolution in runtime
date: 2018-05-25
excerpt: How to remove batch normalization layers from a learned convolutional neural network to speed up/simplify the forward pass.
tags: batch normalization, convolution, PyTorch
layout: default
katex: true
---

We discuss how to simplify the network architecture by merging the freezed batch normalization layer with a preceding convolution.
This is a common setup in practice and deserves to be investigated.

## Introduction and motivation

[Batch normalization](https://arxiv.org/abs/1502.03167) (often abbreviated as BN) is a popular method used in modern neural networks as it often reduces training time and potentially improves generalization
(however, there are some controversies around it: [1](https://www.reddit.com/r/MachineLearning/comments/7issby/d_training_with_batch_normalization/), [2](https://www.reddit.com/r/MachineLearning/comments/67mjuw/d_alternative_interpretation_of/)).

Today's state-of-the-art image classifiers incorporate batch normalization ([ResNets](https://arxiv.org/abs/1512.03385), [DenseNets](https://arxiv.org/abs/1608.06993)).

During runtime (test time, i.e., after training), the functinality of batch normalization is turned off and the approximated per-channel mean $\mu$ and variance $\sigma^2$ are used instead.
This restricted functionality can be implemented as a convolutional layer or, even better, merged with the preceding convolutional layer.
This saves computational resources and simplifies the network architecture at the same time.

## Basics of batch normalization

Let $x$ be a signal (activation) within the network that we want to normalize.
Given a set of such signals ${x_1, x_2, \ldots, x_n}$ coming from processing different samples within a batch,
each is normalized as follows:

$$
	\hat{x}_i = \gamma\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

The values $\mu$ and $\sigma^2$ are the mean and variance computed over a batch, $\epsilon$ is a small constant included for numerical stability, $\gamma$ is the scaling factor and $\beta$ the shift factor.

During training, $\mu$ and $\sigma$ are recomputed for each batch:

$$
	\mu=\frac{1}{n}\sum x_i
$$

$$
	\sigma^2=\frac{1}{n}\sum (x_i - \mu)^2
$$

The parameters $\gamma$ and $\beta$ are slowly learned with gradient descent together with the other parameters of the network.
During test time, we usually do not run the network on a batch of images.
Thus, the previously mentioned formulae for $\mu$ and $\sigma$ cannot be used.
Instead, we use their estimates computed during training by [exponential moving average](https://en.wikipedia.org/wiki/Exponential_smoothing).
Let us denote these approximations as $\hat{\mu}$ nad $\hat{\sigma}^2$.

Nowadays, batch normalization is mostly used in convolutional neural networks for processing images.
In this setting, there are mean and variance estimates, shift and scale parameters for each channel of the input feature map.
We will denote these as $\mu_c$, $\sigma^2_c$, $\gamma_c$ and $\beta_c$ for channel $c$.

## Implementing freezed batchnorm as a $1\times 1$ convolution

Given a feature map $F$ in the $C\times H\times W$ order (channel, height, width), we can obtain its normalized version, $\hat{F}$, by computing the following matrix-vector operations for each spatial position $i, j$:

$$
	\begin{pmatrix}
		\hat{F}_{1,i,j}	\cr[0.5em]
		\hat{F}_{2,i,j}	\cr[0.5em]
		\vdots	\cr[0.5em]
		\hat{F}_{C-1,i,j}	\cr[0.5em]
		\hat{F}_{C,i,j}	\cr[0.5em]
	\end{pmatrix}
	=
	\begin{pmatrix}
		\frac{\gamma_1}{\sqrt{\hat{\sigma}^2_1 + \epsilon}}&0&\cdots&&0\cr[0.5em]
		0&\frac{\gamma_2}{\sqrt{\hat{\sigma}^2_2 + \epsilon}}\cr[0.5em]
		\vdots&&\ddots&&\vdots\cr[0.5em]
		&&&\frac{\gamma_{C-1}}{\sqrt{\hat{\sigma}^2_{C-1} + \epsilon}}&0\cr[0.5em]
		0&&\cdots&0&\frac{\gamma_C}{\sqrt{\hat{\sigma}^2_C + \epsilon}}\cr[0.5em]
	\end{pmatrix}
	\cdot
	\begin{pmatrix}
		F_{1,i,j}	\cr[0.5em]
		F_{2,i,j}	\cr[0.5em]
		\vdots	\cr[0.5em]
		F_{C-1,i,j}	\cr[0.5em]
		F_{C,i,j}	\cr[0.5em]
	\end{pmatrix}
	+
	\begin{pmatrix}
		\beta_1 - \gamma_1\frac{\hat{\mu}_1}{\sqrt{\hat{\sigma}^2_1 + \epsilon}}	\cr[0.5em]
		\beta_2 - \gamma_2\frac{\hat{\mu}_2}{\sqrt{\hat{\sigma}^2_2 + \epsilon}}	\cr[0.5em]
		\vdots	\cr[0.5em]
		\beta_{C-1} - \gamma_{C-1}\frac{\hat{\mu}_{C-1}}{\sqrt{\hat{\sigma}^2_{C-1} + \epsilon}}	\cr[0.5em]
		\beta_C - \gamma_C\frac{\hat{\mu}_C}{\sqrt{\hat{\sigma}^2_C + \epsilon}}	\cr[0.5em]
	\end{pmatrix}
$$

We can see from the above equation that these operations can be implemented in modern deep-learning frameworks as a $1\times 1$ convolution.
Moreover, since the BN layers are often placed after convolutional layers, we can fuse these together.

## Fusing batch normalization with a convolutional layer

Let $\mathbf{W}_{BN}\in\mathbb{R}^{C\times C}$ and $\mathbf{b}_{BN}\in\mathbb{R}^{C}$ denote the matrix and bias from the above equation, and $\mathbf{W}_{conv}\in\mathbb{R}^{C\times(C_{prev}\cdot k^2)}$ and $\mathbf{b}_{conv}\in\mathbb{R}^{C}$ the parameters of the convolutional layer that precedes batch normalization, where $C_{prev}$ is the number of channels of the feature map $F_{prev}$ input to the convolutional layer and $k\times k$ is the filter size.

Given a $k\times k$ neighbourhood of $F_{prev}$ unwrapped into a $k^2\cdot C_{prev}$ vector $\mathbf{f}_{i,j}$, we can write the whole computational process as:

$$
	\mathbf{\hat{f}}_{i,j}=
	\mathbf{W}_{BN}\cdot (\mathbf{W}_{conv}\cdot\mathbf{f}_{i,j} + \mathbf{b}_{conv}) + \mathbf{b}_{BN}
$$

Thus, we can replace these two layers by a single convolutional layer with the following parameters:

* filter weights: $\mathbf{W}=\mathbf{W}_{BN}\cdot \mathbf{W}_{conv}$;
* bias: $\mathbf{b}=\mathbf{W}_{BN}\cdot\mathbf{b}_{conv}+\mathbf{b}_{BN}$.

## Implementation in PyTorch

In Pytorch, each convolutional layer `conv` has the following parameters:

* filter weights, $\mathbf{W}$: `conv.weight`;
* bias, $\mathbf{b}$: `conv.bias`;

and each BN layer `bn` layer has the following ones:

* scaling, $\gamma$: `bn.weight`;
* shift, $\beta$: `bn.bias`;
* mean estiamte, $\hat{\mu}$: `bn.running_mean`;
* variance estimate, $\hat{\sigma}^2$: `bn.running_var`;
* $\epsilon$ (for numerical stability): `bn.eps`.

The following function takes as arguments two PyTorch layers, `nn.Conv2d` and `nn.BatchNorm2d`, and fuses them together into a single `nn.Conv2d` layer.

Edit on October 2021: fix bug found by Pattarawat Chormai ([details](fix.txt)).

```
def fuse_conv_and_bn(conv, bn):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) )
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
	#
	# we're done
	return fusedconv
```

The following code snippet tests the above function on the first two layers of ResNet18:

```
import torch
import torchvision
torch.set_grad_enabled(False)
x = torch.randn(16, 3, 256, 256)
rn18 = torchvision.models.resnet18(pretrained=True)
rn18.eval()
net = torch.nn.Sequential(
	rn18.conv1,
	rn18.bn1
)
y1 = net.forward(x)
fusedconv = fuse_conv_and_bn(net[0], net[1])
y2 = fusedconv.forward(x)
d = (y1 - y2).norm().div(y1.norm()).item()
print("error: %.8f" % d)
```
