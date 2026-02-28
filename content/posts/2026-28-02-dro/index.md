+++
title = "Distributionally Robust Optimization - Part 1"
date = 2026-02-28
author = "Gabriel Stechschulte"
categories = ["optimization", "jax", "probability"]
ShowToc = true
TocOpen = false
draft = false
math = true
+++

In classical Robust Optimization (RO), we are uncertain about which outcome will be realized, so we optimize for the **worst** possible outcome. For example, a mechanical engineer is designing the wings of an airplane such that the weight is minimized, but the wing still has to withstand the stresses under the worst possible conditions.

These worst possible conditions are encoded as uncertainty sets $W$

$$\min_{x \in \mathcal{X}} \max_{w \in \mathcal{W}} F(x, w)$$

Creating these uncertainty sets $W$ can be difficult. Moreover,...

We can do better. In Distributionally Robust Optimization (DRO) we are uncertain about the data generating process, i.e., what distribution generated the data we observe. For example, if there is a random variable (RV) in the objective function or constraints, for example, a forecast, that is distributed according to some unknown distribution. However, we may have collected $N$ historical samples of the outcome and can construct an empirical probability distribution.

$$\hat{\mathbb{P}} = \frac{1}{N} \sum_{i=1}^N \delta_{s_i}$$

Thus, we could use sample average approximation (SAA) and directly optimize against it. But this can lead to poor out-of-sample performance. For example, if there is distribution-shift and the forecast for the next day is completely off with respect to the empirical distribution. DROs idea is to optimize against the worst case distribution in a neighborhood around $\hat{\mathbb{P}$. This neighborhood is called the **ambiguity set** and quantifies the set of distributions within some distance of the empirical distribution.
