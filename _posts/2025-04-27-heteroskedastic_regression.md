---
title: Heteroskedastic Regression
date: 2025-04-27
categories: [Regression]
tags: [parametric, regression, maximum_likelihood_estimation]  
math: true
---

*Heteroskedastic Regression* is a method in machine learning, which not only solves for finding to mean response function but also aims to model the distribution (in terms of the variance) of the residuals as part of the same optimization problem. Recall that generic regression problem can be written as

$$
\begin{equation}
    Y = \mu(X) + \epsilon(X)
\end{equation}
$$

where $$\mu(X)$$, (often also denoted by $$f(X)$$) is the deterministic data generation process and $$\epsilon(X)$$ is the stochastic noise following some (unknown) distribution $$F$$. Without loss of generality we can assume that $$\mathbb{E}[ \epsilon(X)] =0$$, if it was non-zero we could simply absorb it into $$\mu(X)$$. When variance of the errors are indepedent of $$X$$, $$\text{Var}(\epsilon(X)) = \sigma^2$$, we call the errors *homoskedastic*. The complementary case, when the variance depends on $$X$$, i.e. some unknown function $$\sigma^2(X)$$, is called *heteroskedasticity*. Heteroskedastic regression, is a regression problem for both $$\mu(X)$$ and $$\sigma^2(X)$$ simultaneously. 


A different way of thinking about the regression problem is to write $$Y$$ as random variable drawn from a probablitiy distribution $$F$$ with mean $$\mu(X)$$ and variance $$\sigma^2(X)$$ conditionally on $$X$$

$$
\begin{equation}
Y | X \sim F(\mu(X), \sigma^2(X))
\end{equation}
$$

So we can think of the regression problem above as a *Maximum Likelihood Esimtation* problem for which both mean $$\mu(X)$$ and variance $$\sigma^2(X)$$ are unknown. In practice instead of maximizing the likelihood one minimizes the negative log-likelihood (NLL), which is an equivalant problem. Given the likelihood $$\mathcal{L}_F$$ for a distribution $$F$$ (e.g. normal) which can be parameritrized with mean and variace (e.g. a normal distribution or a gamma distribution). We can write the minimization problem as 

$$
\begin{equation}
-\log \mathcal{L}_F(\mu(X),\sigma^2(X);y)
\end{equation}
$$

With this formulation, where the heteroskedasticity of the errors is explictly modelled is also called *Heteroskedastic regression*. 

<!-- Example for a normal distribution, with output multidimensional

$$
\frac{1}{2} \sum_{(x,y)} \log |2\pi \Sigma(x)| + (y-\mu(x))^T \Sigma(x)^{-1} (y-\mu(x))
$$

For independent homoskedastic errors, $$\Sigma ~ I$$, this simplifies to the mean squared error. -->
Assuming a normal distribution, the NLL problem for a 1-d output $$y$$ is

$$
\begin{equation}
    \frac{1}{2} \sum_{(x,y)} \log |2\pi \sigma^2(x)| + \frac{(y-\mu(x))^2}{\sigma(x)^{2}}
    \label{eq:nll}
\end{equation}
$$

For homoskedastic errors, $$\sigma^2(x) = \sigma^2$$, the optimization problem for $$\hat{\mu}$$ is independent of $$\sigma^2$$ and as a result simplifies to minimizing the mean squared error. In the case of heteroskedastic errors, the NLL formulation not only sovles for $$\hat{\sigma}^2(X)$$, it also can be seen as weighted mean square loss problem for $$\hat{\mu}(X)$$ with weights $$w \sim 1/\sigma^2(x)$$. So heteroskedastic regression can actually lead to a different (any typically more efficient) esimate for $$\hat{\mu}$$.


## Prediction intervals

To make a prediction $$\hat{y}=\mu(X_{n+1})$$ the expected uncertainty for the predicion is $$\hat{\sigma}(X_{n+1})$$

The corresponding prediction interval for $$\hat{y}$$ for confidence level $$1-\alpha$$ is 

$$
\begin{equation}
\left[\hat{\mu}(X_{n+1}) - t_{1-\alpha/2, n-p} \sqrt{\hat{\sigma}^2(X_{n+1})},\; \hat{\mu}(X_{n+1}) + t_{1-\alpha/2, n-p} \sqrt{\hat{\sigma}^2(X_{n+1})} \right]
\end{equation}
$$

## Methods for Heteroskedastic Regression

In neural network models, we need to addtional steps. First we need to change the network architecture two output two responses, one for the mean and one for the variance. The second step is modifying the loss function, with one that minimizes the NLL. In PyTorch for one can use https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html

For most model ML models in the sklearn eco-sphere there is not support for a NLL loss function unfortunately. One noticable exception the Boosted Trees algorithm CatBoots which supports a NLL for normally distributed errors https://catboost.ai/docs/en/concepts/loss-functions-regression#RMSEWithUncertainty



<!-- ## Pros and Cons of Heteroskedastic Regression
+ For NN or some (bootest treeds), it is straightfoward to take an existing model and some minor modification to allow for a NLL loss function.
- In practice the errors will not be normally distributed, or is even the familty of distribution known. 
- No statistical garantuees
+ Full distribution from which we can sample
 -->


<!-- ## Normalizing flows -->