---
aliases:
- /probability/2022/05/31/Probabilistic-Predicion-Problems-Part-1
categories:
- probability
date: '2022-05-31'
layout: post
title: Probabilistic Prediction Problems - Part 1

---

Part one is a collection of notes on the problems with probabilistic predictions.

## Distributions over Actions

It is possible to assume the set of possible actions is to pick a class label (or "reject" option) in classification or a real valued scalar as in a point estimate of a parameter. However, it is also possible to assume the set of possible actions is to pick a probability distribution over some value of interest (paramter or class label). That is, we want to perform probabilistic prediction or probabilistic forecasting rather than predicting a scalar. From a decision theoretic approach, we assume the true state of nature is a distribution, $h = p(Y \vert x)$, with the action being another distribution, $a = q(Y \vert x)$. The goal is to be as close to the true state of nature $p$ with our approximation $q$. Therefore, we want to pick $q$ to minimize $\mathbb{E}\ell(p, q)$ for a given $x$.

When making a prediction, the accuracy depends upon the definition of the target, and there is no universal best target. Rather, a decision-theoretic approach should be taken, in particular because of the following two dimensions:

1. **Cost benefit analysis** - What's the risk (or cost) we face when we are wrong? What's the payoff when we are right?
2. **Context** - Some prediction tasks are inheritently harder than others. Therefore, how can we judge how much a model can possibly improve prediction? 

## KL Divergence, Cross-Entropy, and Log-Loss

If we want to compare two distributions, a common loss function is Kullback-Leibler Divergence (KL Divergence).

### Proper Scoring Rules

The key property of proper scoring rules is that the loss function is minimized i.f.f the decision maker picks the distribution $q$ that matches the true distribution $p$. Such a loss function $\ell$ is a _proper scoring rule_.

Maximizing a proper scoring rule will force the model to match the true probabilities, i.e., the probabilities are calibrated. For example, when the weather person states the probability of rain is $70\%$, then it should rain about $70\%$ of the time. 

The following are proper scoring "loss functions":

- Negative log-likelihood - negative of the log-likelihood because _maximization is for losers_
    - log scoring rule - log of the joint probability
- Cross entropy -
    - log-loss 
- Brier score - 


### Information and Uncertainty

We want to use the log probability of the data to score the accuracy of competing models