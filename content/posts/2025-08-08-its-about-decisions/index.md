+++
title = "[WIP] It's About Making Decisions"
date = 2025-08-08
author = "Gabriel Stechschulte"
categories = ["optimization", "probabilistic-programming", "jax"]
ShowToc = true
TocOpen = false  # Optional - keeps TOC open by default
draft = true
math = true
+++

Concepts:
- Data science is largely about making decisions.
- Disappointed with the Bayesian community.
- How PPLs can be a powerful tool for sequential decision making under uncertainty.
- What is sequential decision making?
- Energy modeling demo (part 1).

For companies applying data science and machine learning, it is largely about making decisions—how much inventory should we stock?, how much resources should we allocate?, what parameters should be used for this manufacturing machine?—To arrive at such a decision, analyzing historical data and developing some sort of model is often required. This is the easy part. How this model and analysis is used in downstream systems to make optimal decisions is the hard part. The most impactful data scientists at a company are developing and deploying methods and systems to make decisions, not to develop the next Deep XG Random Effects Boosted Neural Tree Networks&trade; that achieves a super "good" evaluation metric. This post will be the first in a series of posts on sequential decision making under uncertainty. We will use probabilistic programming languages (PPLs) to model uncertainty and then embed these models in a sequential decision making framework to make optimal decisions.

## Sequential decision making

Sequential decision problems can be written as

$$\text{Decision} \rightarrow \text{Information} \rightarrow \text{Decision}$$

Each time we make a decision we incur a cost or recieve a contribution where decisions are made using a *policy*.

## Energy storage

We can use the posterior predictive distribution to evaluate the performance of the policy for varying values of $\theta$. If we want to evaluate the policy using historical data, we can use the sampled site `mu` as this reflects the conditional means

$$\mathbb{E}[y_t|y_{t-1}, y_{t-2}, \theta]$$

The samples from this site will vary due to parameter uncertainty. Evaluating the policy out-of-sample (forecasting).. because it contains stochastic samples from the AR(2) process which includes parameter and stochastic uncertainty.
