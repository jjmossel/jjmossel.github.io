---
title: Sources of uncertainty
date: 2025-04-19
categories: [Introduction]
tags: []  
math: true
---


# Sources of uncertainty
Suppose we have data $$(X,y)$$ which is described by the following relation:

$$
\begin{equation}
y = f(X) + \epsilon
\end{equation}
$$ 

while the *target* $$y$$ and the *features* $$X$$ are observed, the actual model $$f$$ is unknown. $$\epsilon$$ is unobserved noise due to randomness.
The task of Machine learning (ML) is to find a $$\hat{f}$$ given sample data $$\{(X_i,y_i)\}$$ which approximates the unknown $$f$$ as good as possible. 

Once we have found a model $$\hat{f}$$ and make predictions with for a new $$X_n$$ there will be uncertainty in these predictions. The prediction error can be (theortically) decomposed into two terms:

$$
\begin{equation}
\hat{f}(X_n) - y_n = \left(\hat{f}(X_n) - f(X_n)\right) - \epsilon_n
\end{equation}
$$

The first term, $$\left(\hat{f}(X_n) - f(X_n)\right)$$, is the *model uncertainty* and the second term $$\epsilon_n$$ is the *data uncertainty*.

The model uncertainty, is also called epistemic (aka systematic) uncertainty. In theory this uncertainty can be reduced to zero given enough training data and using model family which contains $$f$$.

The second type, data uncertainty, is also called aleatoric (aka statistical) uncertainty. The data uncertainty is doe to the inherit randomness and is irreducible. 

While it is clear that in most ML problems we (indirectly) have to deal with these two sources of uncertainty, it is not always obvious where to draw the line between the two. For example we cannot know for sure that $$f$$ and $$X$$ describe the deterministic part of the problem completly, there could be a (possible unknown) richer feature set $$\tilde{X}$$ and corresponding a function $$\tilde{f}$$ which explain some of the originally assumed data uncertainty.

## Illustration of model and data uncertainty
In this simplified example, we assume that our model family contains the true function $$f$$. When we increase the number of samples we see that the *model uncertainty* gradually goes to zero while the *data uncertainty* and as a result the *total uncertainty* converges to a finite value.

{% include plotly_plots/uncertainty_sources.html%}

## Further Reading
* [*Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods* - E. Hüllermeier & W. Waegeman (2021)](https://link.springer.com/article/10.1007/s10994-021-05946-3)

