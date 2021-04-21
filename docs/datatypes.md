# Data Types

In this setup, we consider the case where we make a prediction about some random variable $Y$. We will focus on two setups: regression where $Y$ takes values in $\mathbb{R}$ (e.g. $Y$ could represent tomorrow's temperature in your city); and classification where $Y$ takes values in some discrete set $\mathcal{Y}$ (e.g. $Y$ could represent the object category of an image).

| Name          | Classification | Regression |  Variable type/shape  |  
| -----------   | -------------  |  --------- |  --------------       |
| Point         | Yes            | Yes        | ``array [batch_size]`` |                       
| Distribution  | Yes            | Yes        | Python class that behaves like ``torch.distribution.Distribution`` |
| Set           | Yes            | No         | ``array [set_size, batch_size]`` | 
| Interval      | No             | Yes        | ``array [2, batch_size]``   | 
| Quantile      | No             | Yes        | ``array [2, num_quantile, batch_size]`` |
| Particle      | Yes            | Yes        | ``array [num_particle, batch_size]``    |

There is no support for multivariate regression yet. Support for multivariate regression will be included in the next major update. 

## Point 

A point prediction is the simplest type of prediction. 

## Particle 

A particle prediction can be represented as a sequence $\mathbb{r} = (r_1, \cdots, r_N)$ where each $r_n$ takes values in $\Yc$. 

Particle predictions typically arise from ensembles, where an panel of experts each make a point prediction about $Y$. 

## Distribution 

## Moment 

## Interval 

## Set

## Quantile 

A quantile prediction can be represented as two sequences of real numbers $\mathbb{r} = (r_1, \cdots, r_N)$ and $\mathbb{\alpha} = (\alpha_1, \cdots, \alpha_N)$ where ideally 

$$
\Pr[Y \leq r_n] = \alpha_n, \forall n=1, \cdots, N
$$

In torchuq a quantile prediction is represented by an array ``[batch_size, N, 2]``. 

A proper scoring rule for a quantile prediction is the pinball loss (also called the hinge loss). 