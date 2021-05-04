# Data Types

In this setup, we consider the case where we make a prediction about some random variable $Y$. We will focus on two setups: regression where $Y$ takes values in $\mathbb{R}$ (e.g. $Y$ could represent tomorrow's temperature in your city); and classification where $Y$ takes values in some discrete set $\mathcal{Y}$ (e.g. $Y$ could represent the object category of an image).

| Name          | Classification | Regression |  Variable type/shape  |  
| -----------   | -------------  |  --------- |  --------------       |
| Point         | Yes            | Yes        | ``array [batch_size]`` |                       
| Distribution  | Yes            | Yes        | Python class that behaves like ``torch.distribution.Distribution`` |
| Set           | Yes            | No         | ``array [set_size, batch_size]`` | 
| Interval      | No             | Yes        | ``array [2, batch_size]``   | 
| Quantile      | No             | Yes        | ``array [batch_size, num_quantile, 2]`` or ``[num_quantile, batch_size]``  |
| Particle      | Yes            | Yes        | ``array [num_particle, batch_size]``    |

There is no support for multivariate regression yet. Support for multivariate regression will be included in the next major update. 

## Point 

A point prediction is the simplest type of prediction. 

## Particle 

A particle prediction can be represented as a sequence $\mathbb{r} = (r_1, \cdots, r_N)$ where each $r_n$ takes values in $\Yc$. 

Particle predictions typically arise from ensembles, where an panel of experts each make a point prediction about $Y$. 

## Distribution 

A distribution prediction is any class that sub-classes (or behaves like) ``torch.distribution.Distribution``. 

Available Examples: 

## Moment 

## Interval 

An interval prediction 

## Set

## Quantile 

A quantile prediction can be represented as two sequences of real numbers $\mathbb{r} = (r_1, \cdots, r_K)$ and $\mathbb{\alpha} = (\alpha_1, \cdots, \alpha_K)$ where ideally 

$$
\Pr[Y \leq r_k] = \alpha_k, \forall k=1, \cdots, K
$$

In torchuq a quantile prediction is represented by an array ``[2, K, batch_size]``. For such an array $x$, $x[0, k, b]$ is the $k$-th quantile value $r_k$ for the $b$-th prediction; $x[1, k, b]$ is the $k$-th quantile $\alpha_k \in [0, 1]$ for the $b$-th prediction. A quantile prediction can also be conveniently represented as an array ``[K, batch_size]``, in which case the quantiles $\mathbb{\alpha}$ are automatically induced as $[1/(K+1), 2/(K+1), \cdots, K/(K+1)]$.   

For available metrics for the quantile representation, see 

Available Examples: conformalized quantile regression 