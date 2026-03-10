+++
title = 'Distributionally Robust Optimization - Part 1'
date = 2026-03-06
author = 'Gabriel Stechschulte'
categories = ['probabilistic-programming', 'optimization']
draft = false
layout = 'marimo'
notebook = '/notebooks/wasserstein_sampling/'
+++

<!--In classical Robust Optimization (RO), we are uncertain about some of the parameters or data and we want the optimal solution to perform "the best possible" assuming that the uncertain data may turn out to be "the worst possible". For example, in a production planning problem, uncertainty affects the duration of tasks and we may optimize for the worst possible duration.

These worst possible conditions are encoded as uncertainty sets $\mathcal{W}(\theta)$ where $/theta$ parameterizes how conservative that set is. The optimizer is then selecting $x$ to minimize the cost where $w \in \mathcal{W}$ is the worst.

Uncertainty sets are the defining feature of RO—and the source of its practical difficulty—is that uncertainty is represented not with a probability distribution, but with a set of plausible outcomes. For example, a simple uncertainty set could be a box constraint

$$
\mathcal{W}(\theta) = \{w|\theta_{i}^{l} \leq w_i \leq \theta_{i}^{u}, \forall i\}
$$-->

In Distributionally Robust Optimization (DRO) we are uncertain about the data generating process, i.e., what distribution generated the data we observe. For example, if there is a random variable (RV) in the objective function or constraints (e.g. a forecast) that is distributed according to some unknown distribution. However, we may have collected $N$ historical samples of the outcome and can construct an empirical probability distribution.

$$\hat{\mathbb{P}} = \frac{1}{N} \sum_{i=1}^N \delta_{s_i}$$

With the these samples, we could use sample average approximation (SAA) and directly optimize against it. But this can lead to poor out-of-sample performance. For example, if there is distribution-shift and the forecast for the next day is completely off with respect to the empirical distribution then the optimization will also be sub-optimal. DROs idea is to optimize against the worst case distribution in a neighborhood around $\hat{\mathbb{P}}$. This neighborhood is called the **ambiguity set** and quantifies the set of distributions within some distance of the empirical distribution.

$$
\mathcal{B}^{\epsilon} := \{\mathbb{Q} \in \mathcal{P}_1(\mathcal{S}) \mid d_W(\hat{\mathbb{P}}, \mathbb{Q}) \leq \epsilon \}
$$

where $\mathcal{B}$ is the collection of all distributions $\mathbb{Q}$ in the set $\mathcal{S}$ whose Wasserstein distance from $\hat{\mathbb{P}}$ is at most $\epsilon$. The Wasserstein ball is the set of distributions we "could reach" within some distance of the empirical distribution if we expend some amount of work.

- Small $\epsilon \rightarrow$ the worst case distribution won't be too different from the empirical distribution.
- Large $\epsilon \rightarrow$ the worst case distribution can be significantly different from the empirical distribution.

Conceptually, this makes sense but it is not so obvious how one would go about implementing such a method. To try and gain some intuition behind the method, lets draw $N$ samples from a Normal distribution that will act as our empirical distribution $\hat{\mathbb{P}}$. Then, we will draw samples from the  Wasserstein ball of radius $\epsilon$ around $\hat{\mathbb{P}}$ to visually see how different these distributions $\mathbb{Q}$ are from $\hat{\mathbb{Q}}$.

In 1D, the Wasserstein distance between two equal-weighted empirical distributions reduces to the average absolute difference between their sorted quantiles.

Putting a ball around $\hat{\mathbb{P}}$ in distribution space reduces to defining a perturbation vector $\delta \in \mathbb{R}^N$ where $\delta_i = y_i - x_i$ is how much you shift the $i$-th sorted quantile.

As we increase $\epsilon$ the perturbations $\delta$ get bigger resulting in a much more different distribution compared to the empirical distribution as indicated by the Wasserstein distance.

Play with the slider below to change the value of $\epsilon$ and observe how $\mathbb{Q}$ changes
