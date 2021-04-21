
Suppose we are given an interval prediction $$[l_i, u_i]$$ such that $$l_i < u_i$$ for samples $i=1, \cdots, n$, define $\mu_i - \frac{u_i - l_i}{2}$ as the mid-point of the predicted interval. We can define the score as 

$$ c_i(y) = (y - \mu_i) / (u_i - l_i) $$ 

This score has the property that $y \in [l_i, u_i]$ if and only if $c_i(y) \in [-1/2, 1/2]$. 