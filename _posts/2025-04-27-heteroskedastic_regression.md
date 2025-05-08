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

where $$\mu(X)$$, (often also denoted by $$f(X)$$) is the deterministic data generation process and $$\epsilon(X)$$ is the stochastic noise following some (unknown) distribution $$F$$. Without loss of generality we can assume that $$\mathbb{E}[ \epsilon(X)] = 0$$, if it was non-zero we could simply absorb it into $$\mu(X)$$. When variance of the errors are independent of $$X$$, $$\text{Var}(\epsilon(X)) = \sigma^2$$, we call the errors *homoskedastic*. The complementary case, when the variance depends on $$X$$, i.e. some unknown function $$\sigma^2(X)$$, is called *heteroskedasticity*. Heteroskedastic regression, is a regression problem for both $$\mu(X)$$ and $$\sigma^2(X)$$ simultaneously. 


A different way of thinking about the regression problem is to write $$Y$$ as random variable drawn from a probability distribution $$F$$ with mean $$\mu(X)$$ and variance $$\sigma^2(X)$$ conditionally on $$X$$

$$
\begin{equation}
Y | X \sim F(\mu(X), \sigma^2(X))
\end{equation}
$$

So we can think of the regression problem above as a *Maximum Likelihood Estimation* problem for which both mean $$\mu(X)$$ and variance $$\sigma^2(X)$$ are unknown. In practice instead of maximizing the likelihood one minimizes the negative log-likelihood (NLL), which is an equivalent problem. Given the likelihood $$\mathcal{L}_F$$ for a distribution $$F$$ (e.g. normal) which can be parametrized with mean and variance (e.g. a normal distribution or a gamma distribution). We can write the minimization problem as 

$$
\begin{equation}
-\log \mathcal{L}_F(\mu(X),\sigma^2(X);y)
\end{equation}
$$

With this formulation, where the heteroskedasticity of the errors is explicitly modelled is also called *Heteroskedastic regression*. 

<!-- Example for a normal distribution, with output multidimensional

$$
\frac{1}{2} \sum_{(x,y)} \log |2\pi \Sigma(x)| + (y-\mu(x))^T \Sigma(x)^{-1} (y-\mu(x))
$$

For independent homoskedastic errors, $$\Sigma ~ I$$, this simplifies to the mean squared error. -->
Assuming a normal distribution, the NLL problem for a 1-d output $$y$$ is

$$
\begin{equation}
    \frac{1}{2} \sum_{(x,y)} \log |2\pi \sigma^2(x)| + \frac{(y-\mu(x))^2}{\sigma^{2}(x)}
    \label{eq:nll}
\end{equation}
$$

For homoskedastic errors, $$\sigma^2(x) = \sigma^2$$, the optimization problem for $$\hat{\mu}$$ is independent of $$\sigma^2$$ and as a result simplifies to minimizing the mean squared error. In the case of heteroskedastic errors, the NLL formulation not only solves for $$\hat{\sigma}^2(X)$$, it also can be seen as weighted mean square loss problem for $$\hat{\mu}(X)$$ with weights $$w \sim 1/\sigma^2(x)$$. So heteroskedastic regression can actually lead to a different estimate for $$\hat{\mu}$$.


## Prediction intervals

To make a prediction $$\hat{y}=\mu(X_{n+1})$$ the expected uncertainty for the predicion is $$\hat{\sigma}(X_{n+1})$$

The corresponding prediction interval for $$\hat{y}$$ for confidence level $$1-\alpha$$ is 

$$
\begin{equation}
\left[\hat{\mu}(X_{n+1}) - t_{1-\alpha/2, n-p} \sqrt{\hat{\sigma}^2(X_{n+1})},\; \hat{\mu}(X_{n+1}) + t_{1-\alpha/2, n-p} \sqrt{\hat{\sigma}^2(X_{n+1})} \right]
\end{equation}
$$


## Heteroskedastic Regression for Deep Learning

In neural network models, we need two additional steps. First we need to change the network architecture two output two responses, one for the mean and one for the variance. A typical architecture is shown below. The first layers are shared, the subsequent layers are bifurcated for the mean and variance outputs respectively. The second step is modifying the loss function, with one that minimizes the NLL. 


![Example Deep Learning for Heteroskedastic Regression](/assets/img/heteroskedastic_regression_deeplearning.svg)

### Defining the Network in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

class HeteroskedasticRegression(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, branch_dim=4):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean_branch = nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, 1)
        )

        self.var_branch = nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        mean = self.mean_branch(shared_out)
        var = torch.exp(self.var_branch(shared_out))  #take exp to ensure positivity
        return mean, var
```

Training the Network

```python
model = HeteroskedasticRegression(hidden_dim=8, branch_dim=4)
loss_fn = nn.GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

for epoch in range(5000):
    model.train()
    optimizer.zero_grad()

    mean_pred, var_pred = model(x_tensor)
    loss = loss_fn(y_tensor, mean_pred, var_pred)

    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    mean_pred, log_var_pred = model(x_tensor)
    std_pred = torch.sqrt(var_pred).numpy()
    mean_pred = mean_pred.numpy()
```

### Example

![Example Deep Learning for Heteroskedastic Regression](/assets/img/heteroskedastic_regression_example.png)

## Faithful Heteroskedastic Regression
A heteroskedastic model is not guaranteed to give the same (or as accurate) predictions as a model which is only trained using the mean squared error.
From \eqref{eq:nll} that is obvious, we see that MSE squared error is weighted by the inverse for the variance $$\sigma^2(x)$$. This has a couple of consequences. Firstly, for $$X$$ for which it's difficult to find the right $$\mu(X)$$ it can increase $$\sigma^2(X)$$ instead to lower the loss. Secondly, when training a model with gradient descent the derivate of of the NLL is inversely proportional to $$\sigma^2(X)$$, resulting in a potentially slower learning for the more high variance regions. In Stirn et al, a modified learning of the NLL has been introduced to address both these issues.


<!-- ## Heteroskedastic Regression with Boosted trees

For most model ML models in the sklearn eco-sphere there is not support for a NLL loss function unfortunately. One noticeable exception the Boosted Trees algorithm CatBoots which supports a NLL for normally distributed errors https://catboost.ai/docs/en/concepts/loss-functions-regression#RMSEWithUncertainty See also https://catboost.ai/docs/en/references/uncertainty  -->

## References


* [*Estimating the mean and variance of the target probability distribution* - D.A. Nix and A.S. Weigend (1994)](https://ieeexplore.ieee.org/document/374138)
* [*Faithful Heteroskedastic Regression with Neural Networks* - A. Stirn et al (2023)](https://proceedings.mlr.press/v206/stirn23a.html)



<!-- ## Pros and Cons of Heteroskedastic Regression
+ For NN or some (boosted trees), it is straightforward to take an existing model and some minor modification to allow for a NLL loss function.
- In practice the errors will not be normally distributed, or is even the family of distribution known. 
- No statistical grantees
+ Full distribution from which we can sample
 -->


<!-- ## Normalizing flows -->