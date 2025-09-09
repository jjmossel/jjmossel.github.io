---
title: Quantile Regression Introduction
date: 2025-04-27
categories: [Regression]
tags: [quantile_regression, non_parameteric]  
math: true
---
Quantile regression is technique that estimates conditionals quantiles. It is a very simple and yet often straightforward way to find conditional quantiles with very few assumptions on the distribution.

Whereas in ordinary regression problems one minimizes the mean squares error, which is the conditional mean, the quantile regression minimizes the so-call *pinball loss*. For a given $$\alpha-$$ percentile the 

$$
\begin{equation}
    \rho_{\alpha}(\hat{y},y) = 
        \begin{cases} 
            \alpha (\hat{y} - y) & \text{if} \; \hat{y} \geq y\\
            (1-\alpha)(y - \hat{y}) & \text{if} \; \hat{y} < y
        \end{cases}
\end{equation}
$$


<!-- Theorem

Asymptotic properties

Uniqueness

Biases

Crossings

Multi-quantile -->

## References

* [*Regression Quantiles* - R. Koenker & G. Bassett, Jr. (1978)](http://www.econ.uiuc.edu/~roger/NAKE/rqs78.pdf)
* [*Quantile and Probability Curves Without Crossing* - V. Chernozhukov, I. Fernandez-Val & A. Galichon  (2007)](https://arxiv.org/abs/0704.3649)