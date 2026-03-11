+++
title = 'Distributionally Robust Optimization - Part 1'
date = 2026-03-06
author = 'Gabriel Stechschulte'
categories = ['probabilistic-programming', 'optimization']
draft = false
ShowToc = true
TocOpen = false
layout = 'marimo'
notebook = '/notebooks/wasserstein_sampling/'
+++

I recently read the paper [Carbon-Aware Computing for Data Centers with Probabilistic Performance Guarantees ](https://ieeexplore.ieee.org/document/11250739) which demonstrated the benefits and power of a Distributionally Robust Optimization method and I wanted to dive deeper into the method. This blog post is a result of the deep dive.

## Introduction

In Stochastic Optimization (SO), we are interested in solutions that account for different possible outcomes of uncertain data. Accounting for this uncertainty requires us to define a probability distribution over the uncertain quantities—a forecast error, a demand signal, a compute load—and then optimize an objective defined with respect to that distribution. The choice of how to use the distribution depends on the application: one might minimize expected cost, guard against a high-percentile loss using Conditional Value-at-Risk (CVaR), or enforce a chance constraint that a capacity limit is violated with probability no greater than some threshold. What all of these approaches share is the assumption that we know, or can accurately estimate, the distribution itself. Distributionally Robust Optimization is the framework we reach for when that assumption breaks down.

In Distributionally Robust Optimization (DRO), we are uncertain not just about the outcome of a random variable (RV), but about the data-generating process that produced the observations we have collected. Suppose there is a RV in our objective function or constraints that is distributed according to some true but unknown distribution $\mathbb{P}$. We have collected $N$ historical observations of this quantity, and from them we can construct an empirical distribution $\hat{\mathbb{P}}$ that places equal probability weight $1/N$ on each observed value:

$$
\begin{equation}
\hat{\mathbb{P}} = \frac{1}{N} \sum_{i=1}^{N} \delta_{s^i}
\end{equation}
$$

where $\delta_{s^i}$ is a Dirac mass centered at the $i$-th historical observation $s^i$.

Given these samples, a natural approach is **Sample Average Approximation (SAA)**: simply treat $\hat{\mathbb{P}}$ as if it were the true distribution $\mathbb{P}$ and optimize directly against your historical values. SAA is easy to implement and works well when $N$ is large and the future resembles the past. The problem is that $\hat{\mathbb{P}}$ places zero probability on any outcome not in your historical data. If tomorrow's data looks nothing like you have ever seen before such as a public holiday, a viral product launch or a seasonal shif. The SAA solution provides no protection whatsoever. The empirical distribution is a point estimate in the space of distributions, and like any point estimate, it carries sampling error that grows more severe as $N$ decreases.

DRO's central idea is to hedge against this uncertainty by optimizing against the **worst-case distribution in a neighborhood around $\hat{\mathbb{P}}$**. This neighborhood is called the **ambiguity set**, and it quantifies the collection of distributions that are plausibly consistent with the data you have observed. Formally, we define the ambiguity set as a Wasserstein ball:

$$
\begin{equation}
\mathcal{B}^{\varepsilon} := \{ \mathbb{Q} \in \mathcal{P}_1(\mathcal{S}) \mid d_W(\hat{\mathbb{P}}, \mathbb{Q}) \leq \varepsilon \}
\end{equation}
$$

where $\mathcal{P}_1(\mathcal{S})$ is the space of probability distributions supported on the feasible set $\mathcal{S}$, and $d_W(\hat{\mathbb{P}}, \mathbb{Q})$ is the Wasserstein-1 distance between $\hat{\mathbb{P}}$ and $\mathbb{Q}$. The Wasserstein distance (aka earth mover's distance) measures the minimum cost required to transport the probability mass of one distribution into the shape of another, where cost accounts for both how much mass is moved **and how far it travels**. A distribution that has the same shape as $\hat{\mathbb{P}}$ but shifted slightly upward is close in Wasserstein distance, because all of the mass only needs to travel a short distance. A distribution with probability mass concentrated in a completely different region is far away, because the mass would need to travel a long distance to reach it.

The radius $\varepsilon$ is the central tuning parameter:

- A **small $\varepsilon$** means the adversary can only choose distributions that look very similar to $\hat{\mathbb{P}}$. The worst-case distribution won't deviate much from your historical data, and the DRO solution will be close to the SAA solution.
- A **large $\varepsilon$** means the adversary can choose distributions that look quite different from $\hat{\mathbb{P}}$, including distributions with mass in regions your data has never visited. The solution will be more conservative but more robust to distributional shift.

The DRO problem then takes the form of a minimax optimization: choose the decision $x$ that minimizes cost under the worst-case distribution in $\mathcal{B}^\varepsilon$:

$$
\begin{equation}
\min_{x \in \mathcal{X}} \sup_{\mathbb{Q} \in \mathcal{B}^\varepsilon} \mathbb{E}^{s \sim \mathbb{Q}}[f(x, s)]
\end{equation}
$$

The $\sup$ over $\mathbb{Q}$ represents an adversary that, after you commit to a decision $x$, picks the distribution within the ambiguity set that makes your decision look as bad as possible.

## Implementation

Conceptually, the ambiguity set makes sense but it is not very intuitive to me. For starters, equation (1) defines an infinite-dimensional set, i.e., there are uncountably many probability distributions within any Wasserstein ball. How does one solve the worst-case optimization over such a set? Moreover, how can we develop an intuition for what the ball actually contains? The goal in the proceeding sections is to develop a geometric intuition for what the ambiguity set contains. Then, in a subsequent post, we will address how we actually solve the optimization problem analytically.

*Note*: All code for this blog post can be viewed or downloaded via the embedded Marimo notebook below.

### Wasserstein distance

For our empirical dataset, we will sample data from a Normal distribution with $\mu=5$ and $\sigma=1$. In 1D, the Wasserstein distance between two equal-weight empirical distributions on $N$ points each reduces to the average absolute difference between their sorted quantiles,

$$
\begin{equation}
W_1(\hat{\mathbb{P}}, \mathbb{Q}) = \frac{1}{N} \sum_{i=1}^{N} \left| x_{(i)} - y_{(i)} \right|
\end{equation}
$$

where $x_{(i)}$ and $y_{(i)}$ are the $i$-th order sorted values of $\hat{\mathbb{P}}$ and $\mathbb{Q}$ respectively. Geometrically, this is the area between the two CDFs. This formula reveals something important. Suppose we parameterize a candidate distribution $\mathbb{Q}$ by a perturbation vector $\delta \in \mathbb{R}^N$, where $\delta_i = y_{(i)} - x_{(i)}$ is the shift applied to the $i$-th sorted quantile of $\hat{\mathbb{P}}$. Then the Wasserstein ball constraint $W_1(\hat{\mathbb{P}}, \mathbb{Q}) \leq \varepsilon$ becomes exactly:

$$
\begin{equation}
\frac{1}{N} \sum_{i=1}^{N} |\delta_i| \leq \varepsilon
\end{equation}
$$

This is an **L1 ball constraint** on $\delta$ in $\mathbb{R}^N$. Placing a Wasserstein ball of radius $\varepsilon$ around $\hat{\mathbb{P}}$ in the infinite-dimensional space of distributions is equivalent, in 1D, to constraining a finite perturbation vector to lie in an L1 ball of radius $\varepsilon$.

### Sampling from the Wasserstein ball

To sample a distribution $\mathbb{Q}$ from $\mathcal{B}^\varepsilon$, we need to generate a perturbation vector $\delta$ that satisfies the L1 constraint above. We will implement the following three steps to achieve this:

1. **Choose a random direction.** Draw $\delta_i \sim \mathcal{N}(0, 1)$ for each $i = 1, \ldots, N$. This gives a random direction in $\mathbb{R}^N$.

2. **Project onto the L1 sphere.** Normalize $\delta$ so that its L1 radius equals exactly $\varepsilon$:

$$\delta \leftarrow \delta \cdot \frac{\varepsilon}{\frac{1}{N}\sum_i |\delta_i|}$$

3. **Pull into the interior.** Multiply by $u \sim \text{Uniform}(0, 1)$ to land somewhere between the origin and the sphere surface:

$$\delta_{\text{final}} = u \cdot \delta$$

The resulting perturbation satisfies $\frac{1}{N}\sum_i |\delta_{\text{final},i}| \leq \varepsilon$ by construction. Adding $\delta_{\text{final}}$ to the sorted quantiles of $\hat{\mathbb{P}}$ and re-sorting gives the support of a valid sample distribution $\mathbb{Q} \in \mathcal{B}^\varepsilon$.

Play with the slider below to change the value of $\epsilon$ and observe how $\mathbb{Q}$ changes.
