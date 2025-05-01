---
title: Linear Regression Example
date: 2025-04-25
categories: [Introduction]
tags: [confidence_interval, prediction_interval, linear_regression]  
math: true
---

To gain some intuition about the sources of uncertainty and how they do depend on the size of our training sample we consider the following linear regression problem

$$
y =  X  \beta + \epsilon
$$

with $$y$$ an $$n-$$dimensional target vector, $$X$$ an $$n \times p$$ feature matrix and $$\epsilon$$ an $$n-$$dimensional vector of random variables repressenting the noise. We make the (classical) Guass-Markov assumptions about the errors $$\epsilon$$:

* **zero mean**: $$\mathbb{E}[\epsilon]=0$$
* **spherical errors** :$$\text{Var}(\epsilon) = \sigma^2 I$$ with $$I$$ the $$n \times n$$ identity matrix.

The Ordinary Least Squares (OLS) estimator has the well known analytical solution

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

To make predictions $$\hat{y}$$ given a (new) $$X$$ we can simply compute $$\hat{y} = X\hat{\beta}$$. The prediction errors made by this model can decomposed into two parts

$$
\begin{equation}
    \hat{y} - y = \underbrace{X \hat{\beta} - X \beta}_{\text{model error}} 
    - \underbrace{X \beta - y}_{\text{data error}}
\end{equation}
$$

The model error is due to the uncertainty in the estimated regression coefficients $$\hat{\beta}$$. The data error $$X \beta - y=\epsilon$$ is due to the stochastic noise. To estimate the model error it is useful to write the approximation error $$\hat{\beta} - \beta$$ in terms of $$\epsilon$$

$$
\begin{align}
\hat{\beta} - \beta &= (X^TX)^{-1}X^Ty - \beta \notag\\
    &= (X^TX)^{-1}X^T (X\beta+\epsilon) - \beta \notag\\ 
    &= (X^TX)^{-1}X^T \epsilon
\end{align}
$$

Using the fact that $$\mathbb{E}[\epsilon]=0$$, it is straightforward to see that our estimated regression coefficients are unbiased (which is garantueed by the Gauss-Markov Theorem):

$$
\mathbb{E}[\hat{\beta} - \beta] = (X^TX)^{-1}X^T \; \mathbb{E}[\epsilon] = 0
$$

To estimate the model uncertainty we first compute the variance $$\hat{\beta}$$

$$
\begin{align}
\text{Var}(\hat{\beta}-\beta) &= \text{Var}( (X^TX)^{-1}X^T \epsilon) \notag\\
&= (X^TX)^{-1}X^T  \text{Var}(\epsilon) X (X^TX)^{-1} \notag\\
&= \sigma^2 (X^TX)^{-1}
\end{align}
$$

In the last step we used $$\text{Var}(\epsilon) = \sigma^2 I$$. Given new samples $$\tilde{X}$$ the model uncertainty is simply

$$
\begin{equation}
    \text{Var}(\tilde{X}\hat{\beta}-\tilde{X}\beta) = \sigma^2 \tilde{X} (X^TX)^{-1} \tilde{X}^T
\end{equation}
$$


The data uncertainty, $$\text{Var}(\epsilon)$$ can be estimated the sample variance of the observed residuals $$\hat{y}_i-y_i$$

$$
\begin{equation}
    \hat{\sigma}^2 = \frac{1}{n-p} \sum_{i=1}^n(\hat{y}_i-y_i)^2
\end{equation}
$$

In the numerator, $$n-p$$ reflects the degrees of freedom, resulting in an unbiased estimate for $$\sigma^2$$.


The total uncertainty for making a prediction is the sum of the two variances

$$
\text{Var}(\hat{y}-y) = \sigma^2 \left(1+\tilde{X} (X^TX)^{-1} \tilde{X}^T\right)
$$




<!-- From this expression we can obtain the *standard error* for the regression coefficients: $$\text{se}(\hat{\beta}_i)=\sqrt{\hat{\sigma}^2 (X^TX)^{-1}_{ii}}$$ -->





## Confidence and Prediction Intervals

When making prediction for new $$\tilde{X}$$ we can construct a confidence interval for the predicted mean predicted response $$\tilde{X}\hat{\beta}$$

$$
\begin{equation}
    \left[\hat{y} \pm t_{1-\alpha/2, n-p} \sqrt{\hat{\sigma}^2 \tilde{X} (X^TX)^{-1} \tilde{X}^T }\right]
    \label{eq:ci_interval}
\end{equation}
$$

with $$t_{1-\alpha/2, n-p}$$ the $$t-$$value. The frequentist interpretation is that when repeating this experiment many times, with probability $$1-\alpha$$ the *confidence interval* contains the model $$\tilde{X}\beta$$. Instead of making a states about the predicted mean response, we can construct a so-called *prediction interval* which is an interval for outcome of the predictions $$\hat{y}_n$$ themselves.

$$
\begin{equation}
    \left[\hat{y} \pm t_{1-\alpha/2, n-p} \sqrt{\hat{\sigma}^2 (1+ \tilde{X} (X^TX)^{-1} \tilde{X}^T) }\right]
    \label{eq:pi_interval}
\end{equation}
$$

## Python Example
The statsmodels library can construct both a confidence interval for the mean prediction and prediction interval for the predictions (they call it a confidence interval for the observations)

```python
import statsmodels.api as sm

model = sm.OLS(y_sample, X_sample)
results = model.fit()

predictions = results.get_prediction(X_test)
prediction_summary = predictions.summary_frame(alpha=0.1)
prediction_summary
```

## Illustration of model and data uncertainty
In this simplified example, we assume that our model family contains the true function $$f$$. When we increase the number of samples we see that the *model uncertainty* respresented by the confidence band $$\eqref{eq:ci_interval}$$ gradually goes to zero while the *data uncertainty* and as a result the *total uncertainty* $$\eqref{eq:pi_interval}$$ converges to a fixed width interval.

{% include plotly_plots/uncertainty_sources.html%}
*Move the slider to see what the effect of increasing the sample size is on the model and data uncertainty.*

## References

* [STAT 501: Regression Methods - Online Course PennState](https://online.stat.psu.edu/stat501/)
* [statsmodel - linear regression](https://www.statsmodels.org/dev/regression.html)


<!-- A confidence interval is statement about the unobserved model parameters.
A prediction interval is about a the outcome of future samples. -->



<!-- 

To quantify the epistemic uncertainty we need an estimate for

$$
\var (\hat{beta}X - \beta X )
$$

and like for the aleatoric uncertainty we need an estimate for

$$
\var (y - \beta X)
$$





# \var( \hat{\beta} \hat{\beta}^T )
# $$ -->



