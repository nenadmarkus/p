---
title: Parametric 3D modeling with GPT-3
date: 2022-09-30
excerpt: With some amount of prompt engineering, we can get something that points in the right direction.
tags: lambdaCAD, GPT-3, prompt engineering
layout: default
katex: false
---

We instruct [GPT-3](https://en.wikipedia.org/wiki/GPT-3) [1] to write a computer program that generates a 3D model of an object.

The results are not good, but are interesting: with some prompt engineering, we can make them point in the right direction.

## Motivation

Due to much hyped results obtained by the MIT team in [solving college-level math problems with OpenAI Codex](https://news.mit.edu/2022/machine-learning-university-math-0803) [2],
it makes us wonder if we can apply a similar approach to 3D parametric modeling.

The main trick is to request a large language model ([OpenAI Codex](https://openai.com/blog/openai-codex/)) to write a computer program instead of a direct answer.
This offloads the computational part to a ("mechanical") programming language like Python and solves just the "intelligence" part.
To illustrate this, please see the example below taken from the paper [2].

Input (prompt) using natural language:

> Using `sympy` find the derivative of the function using the definition of the derivative. `f(x) = (x**2-1)/(2*x-3)`

Result:
```
import sympy as sp
x = sp.Symbol('x')
f = (x**2-1)/(2*x-3)
print(sp.diff(f, x))
```

Output of the program: `2*x/(2*x-3) - 2*(x**2-1)/(2*x-3)**2`

Please see the paper [2] (Figure 3) for more examples.
Some are really impressive and definitely demonstrate a kind of intelligence.

Our experiments will not be with the Codex language model, but with GPT-3 instead.
The differene between these two models is that Codex is specialized for computer code and thus might be more useful for our task.
Unfortunately, access to Codex is limited for now and thus we stick to GPT-3.

## Experiments

We will be using [LambdaCAD](https://nenadmarkus.com/lambda), a free and open-source JavaScript-based 3D CAD tool in which you describe shapes with geometric primitives and their combimantions (the union/intersection/difference operations).

## Conclusion

It would be interesting to see what can we get with finetuning.

OpenAI Codex might turn out to be a more capable tool for the problem of parametric 3D modeling.

## References

[1] The OpenAI team. Language Models are Few-Shot Learners. <https://arxiv.org/abs/2005.14165>, 2020

[2] Drori et al. A Neural Network Solves, Explains, and Generates University Math Problems by Program Synthesis and Few-Shot Learning at Human Level. <https://arxiv.org/abs/2112.15594>, 2021
