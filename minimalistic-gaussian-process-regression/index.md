---
title: A minimalistic implementation of Gaussian process regression
date: 2024-09-30
excerpt: A minimalistic implementation of Gaussian process regression in Python
tags: Gaussian process regression
layout: default
katex: false
---

With rapid advances of AI-powered tools, any programming tips/tricks/suggestions will probably soon become obsolete.
But here goes one more.

```python
def all_pairs_euclid_numpy(A, B):
	sqrA = numpy.broadcast_to(numpy.sum(numpy.power(A, 2), 1).reshape(A.shape[0], 1), (A.shape[0], B.shape[0]))
	sqrB = numpy.broadcast_to(numpy.sum(numpy.power(B, 2), 1).reshape(B.shape[0], 1), (B.shape[0], A.shape[0])).transpose()

	d2 = sqrA - 2*numpy.matmul(A, B.transpose()) + sqrB
	d2[d2 < 0] = 0

	return numpy.sqrt(d2)

def apply_kernel(A, B, l=100.0):
	p = all_pairs_euclid_numpy(A, B)/2.0/l
	return numpy.exp(-p)

def learn_interpolator(features, targets, s=0.0, l=100.0):
	K = apply_kernel(features, features, l=l)
	K = K + s*s*numpy.eye(K.shape[0])

	p = numpy.linalg.solve(K, targets)

	def get_prediction(feats):
		k = apply_kernel(features, feats, l=l)
		preds = numpy.matmul(k.transpose(), p)
		return preds

	return get_prediction
```