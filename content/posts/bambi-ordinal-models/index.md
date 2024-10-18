---
title: 'Ordinal Models in Bambi'
date: 2023-09-29
author: 'Gabriel Stechschulte'
draft: false
categories: ['probabilistic-programming']
---

<!--eofm-->


```python
#| code-fold: true
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import warnings

import bambi as bmb

warnings.filterwarnings("ignore", category=FutureWarning)
```

    WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


# Ordinal Regression

This blog post is a copy of the ordinal models documentation I wrote for [Bambi](https://bambinos.github.io/bambi/). The original post can be found [here](https://bambinos.github.io/bambi/notebooks/).

In some scenarios, the response variable is discrete, like a count, and ordered. Common examples of such data come from questionnaires where the respondent is asked to rate a product, service, or experience on a scale. This scale is often referred to as a [Likert scale](https://en.wikipedia.org/wiki/Likert_scale). For example, a five-level Likert scale could be:

- 1 = Strongly disagree
- 2 = Disagree
- 3 = Neither agree nor disagree
- 4 = Agree
- 5 = Strongly agree

The result is a set of **ordered categories** where each category has an associated numeric value (1-5). However, you can't compute a meaningful difference between the categories. Moreover, the response variable can also be a count where meaningful differences can be computed. For example, a restaurant can be rated on a scale of 1-5 stars where 1 is the worst and 5 is the best. Yes, you can compute the difference between 1 and 2 stars, but it is often treated as ordinal in an applied setting.

Ordinal data presents three challenges when modelling:

1. Unlike a count, the differences in the values are not necessarily equidistant or meaningful. For example, computing the difference between "Strongly disagree" and "Disagree". Or, in the case of the restaurant rating, it may be much harder for a restuarant to go from 4 to 5 stars than from 2 to 3 stars. 
2. The distribution of ordinal responses may be nonnormal as the response is not continuous; particularly if larger response levels are infrequently chosen compared to lower ones.
3. The variances of the unobserved variables that underlie the observed ordered category may differ between the category, time points, etc. 

Thus, treating ordered categories as continuous is not appropriate. To this extent, Bambi supports two classes of ordinal regression models: (1) cumulative, and (2) sequential. Below, it is demonstrated how to fit these two models using Bambi to overcome the challenges of ordered category response data.

## Cumulative model

A cumulative model assumes that the observed ordinal variable $Y$ originates from the "categorization" of a latent continuous variable $Z$. To model the categorization process, the model assumes that there are $K$ thresholds (or cutpoints) $\tau_k$ that partition $Z$ into $K+1$ observable, ordered categories of $Y$. The subscript $k$ in $\tau_k$ is an index that associates that threshold to a particular category $k$. For example, if the response has three categories such as "disagree", "neither agree nor disagree", and "agree", then there are two thresholds $\tau_1$ and $\tau_2$ that partition $Z$ into $K+1 = 3$ categories. Additionally, if we assume $Z$ to have a certain distribution (e.g., Normal) with a cumulative distribution function $F$, the probability of $Y$ being equal to category $k$ is

$$P(Y = k) = F(\tau_k) - F(\tau_{k-1})$$

where $F(\tau)$ is a cumulative probability. For example, suppose we are interested in the probability of each category stated above, and have two thresholds $\tau_1 = -1, \tau_2 = 1$ for the three categories. Additionally, if we assume $Z$ to be normally distributed with $\sigma = 1$ and a cumulative distribution function $\Phi$ then

$$P(Y = 1) = \Phi(\tau_1) = \Phi(-1)$$

$$P(Y = 2) = \Phi(\tau_2) - \Phi(\tau_1) = \Phi(1) - \Phi(-1)$$

$$P(Y = 3) = 1 - \Phi(\tau_2) = 1 - \Phi(1)$$


But how to set the values of the thresholds? By default, Bambi uses a Normal distribution with a grid of evenly spaced $\mu$ that depends on the number of response levels as the prior for the thresholds. Additionally, since the thresholds need to be orderd, Bambi applies a transformation to the values such that the order is preserved. Furthermore, the model specification for ordinal regression typically transforms the cumulative probabilities using the log-cumulative-odds (logit) transformation. Therefore, the learned parameters for the thresholds $\tau$ will be logits.

Lastly, as each $F(\tau)$ implies a cumulative probability for each category, the largest response level always has a cumulative probability of 1. Thus, we effectively do not need a parameter for it due to the law of total probability. For example, for three response values, we only need two thresholds as two thresholds partition $Z$ into $K+1$ categories.

### The moral intuition dataset

To illustrate an cumulative ordinal model, we will model data from a series of experiments conducted by philsophers (this example comes from Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)). The experiments aim to collect empirical evidence relevant to debates about moral intuition, the forms of reasoning through which people develop judgments about the moral goodness and badness of actions. 

In the dataset there are 12 columns and 9930 rows, comprising data for 331 unique individuals. The response we are interested in `response`, is an integer from 1 to 7 indicating how morally permissible the participant found the action to be taken (or not) in the story. The predictors are as follows:

- `action`: a factor with levels 0 and 1 where 1 indicates that the story contained "harm caused by action is morally worse than equivalent harm caused by omission".
- `intention`: a factor with levels 0 and 1 where 1 indicates that the story contained "harm intended as the means to a goal is morally worse than equivalent harm foreseen as the side effect of a goal".
- `contact`: a factor with levels 0 and 1 where 1 indicates that the story contained "using physical contact to cause harm to a victim is morally worse than causing equivalent harm to a victim without using physical contact".


```python
trolly = pd.read_csv("https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Trolley.csv", sep=";")
trolly = trolly[["response", "action", "intention", "contact"]]
trolly["action"] = pd.Categorical(trolly["action"], ordered=False)
trolly["intention"] = pd.Categorical(trolly["intention"], ordered=False)
trolly["contact"] = pd.Categorical(trolly["contact"], ordered=False)
trolly["response"] = pd.Categorical(trolly["response"], ordered=True)
```


```python
# 7 ordered categories from 1-7
trolly.response.unique()
```




    [4, 3, 5, 2, 1, 7, 6]
    Categories (7, int64): [1 < 2 < 3 < 4 < 5 < 6 < 7]



### Intercept only model

Before we fit a model with predictors, let's attempt to recover the parameters of an ordinal model using only the thresholds to get a feel for the cumulative family. Traditionally, in Bambi if we wanted to recover the parameters of the likelihood, we would use an intercept only model and write the formula as `response ~ 1` where `1` indicates to include the intercept. However, in the case of ordinal regression, the thresholds "take the place" of the intercept. Thus, we can write the formula as `response ~ 0` to indicate that we do not want to include an intercept. To fit a cumulative ordinal model, we pass `family="cumulative"`. To compare the thresholds only model, we compute the empirical log-cumulative-odds of the categories directly from the data below and generate a bar plot of the response probabilities.


```python
pr_k = trolly.response.value_counts().sort_index().values / trolly.shape[0]
cum_pr_k = np.cumsum(pr_k)
logit_func = lambda x: np.log(x / (1 - x))
cum_logit = logit_func(cum_pr_k)
cum_logit
```

    /var/folders/rl/y69t95y51g90tvd6gjzzs59h0000gn/T/ipykernel_22293/1548491577.py:3: RuntimeWarning: invalid value encountered in log
      logit_func = lambda x: np.log(x / (1 - x))





    array([-1.91609116, -1.26660559, -0.718634  ,  0.24778573,  0.88986365,
            1.76938091,         nan])




```python
plt.figure(figsize=(7, 3))
plt.bar(np.arange(1, 8), pr_k)
plt.ylabel("Probability")
plt.xlabel("Response")
plt.title("Empirical probability of each response category");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_9_0.png)
    



```python
model = bmb.Model("response ~ 0", data=trolly, family="cumulative")
idata = model.fit(random_seed=1234)
```

Below, the components of the model are outputed. Notice how the thresholds are a grid of six values ranging from -2 to 2. 


```python
model
```




           Formula: response ~ 0 + action + intention + contact + action:intention + contact:intention
            Family: cumulative
              Link: p = logit
      Observations: 9930
            Priors: 
        target = p
            Common-level effects
                action ~ Normal(mu: 0.0, sigma: 5.045)
                intention ~ Normal(mu: 0.0, sigma: 5.0111)
                contact ~ Normal(mu: 0.0, sigma: 6.25)
                action:intention ~ Normal(mu: 0.0, sigma: 6.7082)
                contact:intention ~ Normal(mu: 0.0, sigma: 8.3333)
            
            Auxiliary parameters
                threshold ~ Normal(mu: [-2.  -1.2 -0.4  0.4  1.2  2. ], sigma: 1.0, transform: ordered)
    ------
    * To see a plot of the priors call the .plot_priors() method.
    * To see a summary or plot of the posterior pass the object returned by .fit() to az.summary() or az.plot_trace()




```python
az.summary(idata)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>response_threshold[0]</th>
      <td>-1.917</td>
      <td>0.030</td>
      <td>-1.974</td>
      <td>-1.863</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4097.0</td>
      <td>3163.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>response_threshold[1]</th>
      <td>-1.267</td>
      <td>0.024</td>
      <td>-1.312</td>
      <td>-1.220</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5391.0</td>
      <td>3302.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>response_threshold[2]</th>
      <td>-0.719</td>
      <td>0.021</td>
      <td>-0.760</td>
      <td>-0.681</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5439.0</td>
      <td>3698.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>response_threshold[3]</th>
      <td>0.248</td>
      <td>0.020</td>
      <td>0.211</td>
      <td>0.287</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5416.0</td>
      <td>3644.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>response_threshold[4]</th>
      <td>0.890</td>
      <td>0.022</td>
      <td>0.847</td>
      <td>0.930</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4966.0</td>
      <td>3439.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>response_threshold[5]</th>
      <td>1.770</td>
      <td>0.027</td>
      <td>1.721</td>
      <td>1.823</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4785.0</td>
      <td>3368.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Viewing the summary dataframe, we see a total of six  `response_threshold` coefficients. Why six? Remember, we get the last parameter for free. Since there are seven categories, we only need six cutpoints. The index (using zero based indexing) of the `response_threshold` indicates the category that the threshold is associated with. Comparing to the empirical log-cumulative-odds computation above, the mean of the posterior distribution for each category is close to the empirical value.

As the the log cumulative link is used, we need to apply the inverse of the logit function to transform back to cumulative probabilities. Below, we plot the cumulative probabilities for each category. 


```python
expit_func = lambda x: 1 / (1 + np.exp(-x))
cumprobs = expit_func(idata.posterior.response_threshold).mean(("chain", "draw"))
cumprobs = np.append(cumprobs, 1)

plt.figure(figsize=(7, 3))
plt.plot(sorted(trolly.response.unique()), cumprobs, marker='o')
plt.ylabel("Cumulative probability")
plt.xlabel("Response category")
plt.title("Cumulative probabilities of response categories");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_15_0.png)
    



```python
fig, ax = plt.subplots(figsize=(7, 3))
for i in range(6):
    outcome = expit_func(idata.posterior.response_threshold).sel(response_threshold_dim=i).to_numpy().flatten()
    ax.hist(outcome, bins=15, alpha=0.5, label=f"Category: {i}")
ax.set_xlabel("Probability")
ax.set_ylabel("Count")
ax.set_title("Cumulative Probability by Category")
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_16_0.png)
    


We can take the derivative of the cumulative probabilities to get the posterior probabilities for each category. Notice how the posterior probabilities in the barplot below are close to the empirical probabilities in barplot above.


```python
# derivative
ddx = np.diff(cumprobs)
probs = np.insert(ddx, 0, cumprobs[0])

plt.figure(figsize=(7, 3))
plt.bar(sorted(trolly.response.unique()), probs)
plt.ylabel("Probability")
plt.xlabel("Response category")
plt.title("Posterior Probability of each response category");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_18_0.png)
    


Notice in the plots above, the jump in probability from category 3 to 4. Additionally, the estimates of the coefficients is precise for each category. Now that we have an understanding how the cumulative link function is applied to produce ordered cumulative outcomes, we will add predictors to the model. 

### Adding predictors

In the cumulative model described above, adding predictors was explicitly left out. In this section, it is described how predictors are added to ordinal cumulative models. When adding predictor variables, what we would like is for any predictor, as it increases, predictions are moved progressively (increased) through the categories in sequence. A linear regression is formed for $Z$ by adding a predictor term $\eta$

$$\eta = \beta_1 x_1 + \beta_2 x_2 +, . . ., \beta_n x_n$$

Notice how similar this looks to an ordinary linear model. However, there is no intercept or error term. This is because the intercept is replaced by the threshold $\tau$ and the error term $\epsilon$ is added seperately to obtain

$$Z = \eta + \epsilon$$ 

Putting the predictor term together with the thresholds and cumulative distribution function, we obtain the probability of $Y$ being equal to a category $k$ as

$$Pr(Y = k | \eta) = F(\tau_k - \eta) - F(\tau_{k-1} - \eta)$$

The same predictor term $\eta$ is subtracted from each threshold because if we decrease the log-cumulative-odds of every outcome value $k$ below the maximum, this shifts probability mass upwards towards higher outcome values. Thus, positive $\beta$ values correspond to increasing $x$, which is associated with an increase in the mean response $Y$. The parameters to be estimated from the model are the thresholds $\tau$ and the predictor terms $\eta$ coefficients.  

To add predictors for ordinal models in Bambi, we continue to use the formula interface.


```python
model = bmb.Model(
    "response ~ 0 + action + intention + contact + action:intention + contact:intention", 
    data=trolly, 
    family="cumulative"
)
idata = model.fit(random_seed=1234)
```

In the summary dataframe below, we only select the predictor variables as the thresholds are not of interest at the moment.

In the summary dataframe below, we only select the predictor variables as the cutpoints are not of interest at the moment.


```python
az.summary(
    idata, 
    var_names=["action", "intention", "contact", 
               "action:intention", "contact:intention"]
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>action[1]</th>
      <td>-0.466</td>
      <td>0.055</td>
      <td>-0.563</td>
      <td>-0.363</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>412.0</td>
      <td>645.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>intention[1]</th>
      <td>-0.278</td>
      <td>0.060</td>
      <td>-0.390</td>
      <td>-0.167</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>379.0</td>
      <td>500.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>contact[1]</th>
      <td>-0.327</td>
      <td>0.072</td>
      <td>-0.460</td>
      <td>-0.204</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>525.0</td>
      <td>688.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>action:intention[1, 1]</th>
      <td>-0.450</td>
      <td>0.080</td>
      <td>-0.609</td>
      <td>-0.300</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>396.0</td>
      <td>479.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>contact:intention[1, 1]</th>
      <td>-1.278</td>
      <td>0.097</td>
      <td>-1.459</td>
      <td>-1.098</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>557.0</td>
      <td>567.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



The posterior distribution of the slopes are all negative indicating that each of these story features reduces the rating—the acceptability of the story. Below, a forest plot is used to make this insight more clear.


```python
az.plot_forest(
    idata,
    combined=True,
    var_names=["action", "intention", "contact", 
               "action:intention", "contact:intention"],
    figsize=(7, 3),
    textsize=11
);
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_26_0.png)
    


Again, we can plot the cumulative probability of each category. Compared to the same plot above, notice how most of the category probabilities have been shifted to the left. Additionally, there is more uncertainty for category 3, 4, and 5.


```python
fig, ax = plt.subplots(figsize=(7, 3))
for i in range(6):
    outcome = expit_func(idata.posterior.response_threshold).sel(response_threshold_dim=i).to_numpy().flatten()
    ax.hist(outcome, bins=15, alpha=0.5, label=f"Category: {i}")
ax.set_xlabel("Probability")
ax.set_ylabel("Count")
ax.set_title("Cumulative Probability by Category")
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_28_0.png)
    


### Posterior predictive distribution

To get a sense of how well the ordinal model fits the data, we can plot samples from the posterior predictive distribution. To plot the samples, a utility function is defined below to assist in the plotting of discrete values.


```python
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1] * amount, c[2])

def plot_ppc_discrete(idata, bins, ax):
    
    def add_discrete_bands(x, lower, upper, ax, **kwargs):
        for i, (l, u) in enumerate(zip(lower, upper)):
            s = slice(i, i + 2)
            ax.fill_between(x[s], [l, l], [u, u], **kwargs)

    var_name = list(idata.observed_data.data_vars)[0]
    y_obs = idata.observed_data[var_name].to_numpy()
    
    counts_list = []
    for draw_values in az.extract(idata, "posterior_predictive")[var_name].to_numpy().T:
        counts, _ = np.histogram(draw_values, bins=bins)
        counts_list.append(counts)
    counts_arr = np.stack(counts_list)

    qts_90 = np.quantile(counts_arr, (0.05, 0.95), axis=0)
    qts_70 = np.quantile(counts_arr, (0.15, 0.85), axis=0)
    qts_50 = np.quantile(counts_arr, (0.25, 0.75), axis=0)
    qts_30 = np.quantile(counts_arr, (0.35, 0.65), axis=0)
    median = np.quantile(counts_arr, 0.5, axis=0)

    colors = [adjust_lightness("C0", x) for x in [1.8, 1.6, 1.4, 1.2, 0.9]]

    add_discrete_bands(bins, qts_90[0], qts_90[1], ax=ax, color=colors[0])
    add_discrete_bands(bins, qts_70[0], qts_70[1], ax=ax, color=colors[1])
    add_discrete_bands(bins, qts_50[0], qts_50[1], ax=ax, color=colors[2])
    add_discrete_bands(bins, qts_30[0], qts_30[1], ax=ax, color=colors[3])

    
    ax.step(bins[:-1], median, color=colors[4], lw=2, where="post")
    ax.hist(y_obs, bins=bins, histtype="step", lw=2, color="black", align="mid")
    handles = [
        Line2D([], [], label="Observed data", color="black", lw=2),
        Line2D([], [], label="Posterior predictive median", color=colors[4], lw=2)
    ]
    ax.legend(handles=handles)
    return ax
```


```python
idata_pps = model.predict(idata=idata, kind="pps", inplace=False)

bins = np.arange(7)
fig, ax = plt.subplots(figsize=(7, 3))
ax = plot_ppc_discrete(idata_pps, bins, ax)
ax.set_xlabel("Response category")
ax.set_ylabel("Count")
ax.set_title("Cumulative model - Posterior Predictive Distribution");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_31_0.png)
    


## Sequential Model

For some ordinal variables, the assumption of a **single** underlying continuous variable (as in cumulative models) may not be appropriate. If the response can be understood as being the result of a sequential process, such that a higher response category is possible only after all lower categories are achieved, then a sequential model may be more appropriate than a cumulative model.

Sequential models assume that for **every** category $k$ there is a latent continuous variable $Z$ that determines the transition between categories $k$ and $k+1$. Now, a threshold $\tau$ belongs to each latent process. If there are 3 categories, then there are 3 latent processes. If $Z_k$ is greater than the threshold $\tau_k$, the sequential process continues, otherwise it stops at category $k$. As with the cumulative model, we assume a distribution for $Z_k$ with a cumulative distribution function $F$.

As an example, lets suppose we are interested in modeling the probability a boxer makes it to round 3. This implies that the particular boxer in question survived round 1 $Z_1 > \tau_1$ , 2 $Z_2 > \tau_2$, and 3 $Z_3 > \tau_3$. This can be written as 

$$Pr(Y = 3) = (1 - P(Z_1 \leq \tau_1)) * (1 - P(Z_2 \leq \tau_2)) * P(Z_3 \leq \tau_3)$$

As in the cumulative model above, if we assume $Y$ to be normally distributed with the thresholds $\tau_1 = -1, \tau_2 = 0, \tau_3 = 1$ and cumulative distribution function $\Phi$ then

$$Pr(Y = 3) = (1 - \Phi(\tau_1)) * (1 - \Phi(\tau_2)) * \Phi(\tau_3)$$

To add predictors to this sequential model, we follow the same specification in the _Adding Predictors_ section above. Thus, the sequential model with predictor terms becomes

$$P(Y = k) = F(\tau_k - \eta) * \prod_{j=1}^{k-1}{(1 - F(\tau_j - \eta))}$$

Thus, the probability that $Y$ is equal to category $k$ is equal to the probability that it did not fall in one of the former categories $1: k-1$ multiplied by the probability that the sequential process stopped at $k$ rather than continuing past it.

### Human resources attrition dataset

To illustrate an sequential model with a stopping ratio link function, we will use data from the IBM human resources employee attrition and performance [dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). The original dataset contains 1470 rows and 35 columns. However, our goal is to model the total working years of employees using age as a predictor. This data lends itself to a sequential model as the response, total working years, is a sequential process. In order to have 10 years of working experience, it is necessarily true that the employee had 9 years of working experience. Additionally, age is choosen as a predictor as it is positively correlated with total working years.


```python
attrition = pd.read_csv("data/hr_employee_attrition.tsv.txt", sep="\t")
attrition = attrition[attrition["Attrition"] == "No"]
attrition["YearsAtCompany"] = pd.Categorical(attrition["YearsAtCompany"], ordered=True)
attrition[["YearsAtCompany", "Age"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YearsAtCompany</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>32</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>



Below, the empirical probabilities of the response categories are computed. Employees are most likely to stay at the company between 1 and 10 years.


```python
pr_k = attrition.YearsAtCompany.value_counts().sort_index().values / attrition.shape[0]

plt.figure(figsize=(7, 3))
plt.bar(np.arange(0, 36), pr_k)
plt.xlabel("Response category")
plt.ylabel("Probability")
plt.title("Empirical probability of each response category");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_36_0.png)
    


### Default prior of thresholds

Before we fit the sequential model, it's worth mentioning that the default priors for the thresholds in a sequential model are different than the cumulative model. In the cumulative model, the default prior for the thresholds is a Normal distribution with a grid of evenly spaced $\mu$ where an ordered transformation is applied to ensure the ordering of the values. However, in the sequential model, the ordering of the thresholds does not matter. Thus, the default prior for the thresholds is a Normal distribution with a zero $\mu$ vector of length $k - 1$ where $k$ is the number of response levels. Refer to the [getting started](https://bambinos.github.io/bambi/notebooks/getting_started.html#specifying-priors) docs if you need a refresher on priors in Bambi. 

Subsequently, fitting a sequential model is similar to fitting a cumulative model. The only difference is that we pass `family="sratio"` to the `bambi.Model` constructor. 


```python
sequence_model = bmb.Model(
    "YearsAtCompany ~ 0 + TotalWorkingYears", 
    data=attrition, 
    family="sratio"
)
sequence_idata = sequence_model.fit(random_seed=1234)
```


```python
sequence_model
```




           Formula: YearsAtCompany ~ 0 + TotalWorkingYears
            Family: sratio
              Link: p = logit
      Observations: 1233
            Priors: 
        target = p
            Common-level effects
                TotalWorkingYears ~ Normal(mu: 0.0, sigma: 0.3223)
            
            Auxiliary parameters
                threshold ~ Normal(mu: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], sigma: 1.0)




```python
az.summary(sequence_idata)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>YearsAtCompany_threshold[0]</th>
      <td>-2.525</td>
      <td>0.193</td>
      <td>-2.896</td>
      <td>-2.191</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>2027.0</td>
      <td>812.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[1]</th>
      <td>-1.057</td>
      <td>0.110</td>
      <td>-1.265</td>
      <td>-0.850</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1137.0</td>
      <td>601.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[2]</th>
      <td>-1.017</td>
      <td>0.119</td>
      <td>-1.238</td>
      <td>-0.792</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1009.0</td>
      <td>705.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[3]</th>
      <td>-0.760</td>
      <td>0.117</td>
      <td>-0.984</td>
      <td>-0.553</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1210.0</td>
      <td>833.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[4]</th>
      <td>-0.754</td>
      <td>0.126</td>
      <td>-0.969</td>
      <td>-0.493</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1327.0</td>
      <td>780.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[5]</th>
      <td>0.251</td>
      <td>0.111</td>
      <td>0.053</td>
      <td>0.466</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1207.0</td>
      <td>808.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[6]</th>
      <td>-0.500</td>
      <td>0.145</td>
      <td>-0.769</td>
      <td>-0.230</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1394.0</td>
      <td>526.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[7]</th>
      <td>-0.085</td>
      <td>0.137</td>
      <td>-0.355</td>
      <td>0.155</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1115.0</td>
      <td>835.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[8]</th>
      <td>0.025</td>
      <td>0.141</td>
      <td>-0.213</td>
      <td>0.310</td>
      <td>0.004</td>
      <td>0.004</td>
      <td>1081.0</td>
      <td>750.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[9]</th>
      <td>0.389</td>
      <td>0.146</td>
      <td>0.128</td>
      <td>0.673</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1134.0</td>
      <td>865.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[10]</th>
      <td>1.267</td>
      <td>0.154</td>
      <td>0.970</td>
      <td>1.540</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>923.0</td>
      <td>737.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[11]</th>
      <td>0.436</td>
      <td>0.213</td>
      <td>0.055</td>
      <td>0.826</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1376.0</td>
      <td>662.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[12]</th>
      <td>-0.116</td>
      <td>0.307</td>
      <td>-0.662</td>
      <td>0.457</td>
      <td>0.008</td>
      <td>0.010</td>
      <td>1568.0</td>
      <td>693.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[13]</th>
      <td>0.513</td>
      <td>0.250</td>
      <td>0.024</td>
      <td>0.943</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>1537.0</td>
      <td>738.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[14]</th>
      <td>0.392</td>
      <td>0.293</td>
      <td>-0.126</td>
      <td>0.958</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>1328.0</td>
      <td>696.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[15]</th>
      <td>0.791</td>
      <td>0.277</td>
      <td>0.310</td>
      <td>1.348</td>
      <td>0.008</td>
      <td>0.005</td>
      <td>1405.0</td>
      <td>619.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[16]</th>
      <td>0.433</td>
      <td>0.329</td>
      <td>-0.196</td>
      <td>1.017</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>1516.0</td>
      <td>691.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[17]</th>
      <td>0.252</td>
      <td>0.365</td>
      <td>-0.397</td>
      <td>0.916</td>
      <td>0.009</td>
      <td>0.009</td>
      <td>1468.0</td>
      <td>841.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[18]</th>
      <td>0.791</td>
      <td>0.330</td>
      <td>0.160</td>
      <td>1.394</td>
      <td>0.009</td>
      <td>0.007</td>
      <td>1274.0</td>
      <td>856.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[19]</th>
      <td>0.788</td>
      <td>0.347</td>
      <td>0.148</td>
      <td>1.445</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>1306.0</td>
      <td>612.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[20]</th>
      <td>2.191</td>
      <td>0.268</td>
      <td>1.684</td>
      <td>2.680</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>1076.0</td>
      <td>720.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[21]</th>
      <td>1.842</td>
      <td>0.331</td>
      <td>1.323</td>
      <td>2.516</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>1447.0</td>
      <td>825.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[22]</th>
      <td>2.431</td>
      <td>0.369</td>
      <td>1.746</td>
      <td>3.078</td>
      <td>0.011</td>
      <td>0.008</td>
      <td>1230.0</td>
      <td>616.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[23]</th>
      <td>0.017</td>
      <td>0.743</td>
      <td>-1.389</td>
      <td>1.340</td>
      <td>0.016</td>
      <td>0.027</td>
      <td>2194.0</td>
      <td>712.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[24]</th>
      <td>1.581</td>
      <td>0.571</td>
      <td>0.489</td>
      <td>2.582</td>
      <td>0.015</td>
      <td>0.010</td>
      <td>1671.0</td>
      <td>636.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[25]</th>
      <td>1.530</td>
      <td>0.579</td>
      <td>0.491</td>
      <td>2.646</td>
      <td>0.013</td>
      <td>0.010</td>
      <td>2056.0</td>
      <td>616.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[26]</th>
      <td>1.727</td>
      <td>0.607</td>
      <td>0.605</td>
      <td>2.844</td>
      <td>0.014</td>
      <td>0.010</td>
      <td>1982.0</td>
      <td>709.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[27]</th>
      <td>1.096</td>
      <td>0.732</td>
      <td>-0.351</td>
      <td>2.412</td>
      <td>0.017</td>
      <td>0.016</td>
      <td>1855.0</td>
      <td>791.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[28]</th>
      <td>1.156</td>
      <td>0.722</td>
      <td>-0.288</td>
      <td>2.399</td>
      <td>0.016</td>
      <td>0.013</td>
      <td>2073.0</td>
      <td>852.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[29]</th>
      <td>0.541</td>
      <td>0.868</td>
      <td>-1.149</td>
      <td>2.239</td>
      <td>0.019</td>
      <td>0.027</td>
      <td>2055.0</td>
      <td>669.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[30]</th>
      <td>1.247</td>
      <td>0.808</td>
      <td>-0.275</td>
      <td>2.744</td>
      <td>0.019</td>
      <td>0.015</td>
      <td>1821.0</td>
      <td>653.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[31]</th>
      <td>1.385</td>
      <td>0.813</td>
      <td>0.013</td>
      <td>3.021</td>
      <td>0.019</td>
      <td>0.015</td>
      <td>1875.0</td>
      <td>599.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[32]</th>
      <td>2.683</td>
      <td>0.710</td>
      <td>1.318</td>
      <td>3.936</td>
      <td>0.017</td>
      <td>0.012</td>
      <td>1698.0</td>
      <td>815.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[33]</th>
      <td>0.868</td>
      <td>1.018</td>
      <td>-0.974</td>
      <td>2.944</td>
      <td>0.026</td>
      <td>0.024</td>
      <td>1573.0</td>
      <td>781.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>YearsAtCompany_threshold[34]</th>
      <td>1.756</td>
      <td>0.853</td>
      <td>0.348</td>
      <td>3.500</td>
      <td>0.021</td>
      <td>0.017</td>
      <td>1640.0</td>
      <td>876.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>TotalWorkingYears</th>
      <td>0.127</td>
      <td>0.005</td>
      <td>0.118</td>
      <td>0.138</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>482.0</td>
      <td>514.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</div>



The coefficients are still on the logits scale, so we need to apply the inverse of the logit function to transform back to probabilities. Below, we plot the probabilities for each category.


```python
probs = expit_func(sequence_idata.posterior.YearsAtCompany_threshold).mean(("chain", "draw"))
probs = np.append(probs, 1)

plt.figure(figsize=(7, 3))
plt.plot(sorted(attrition.YearsAtCompany.unique()), probs, marker='o')
plt.ylabel("Probability")
plt.xlabel("Response category");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_42_0.png)
    


This plot can seem confusing at first. Remember, the sequential model is a product of probabilities, i.e., the probability that $Y$ is equal to category $k$ is equal to the probability that it did not fall in one of the former categories $1: k-1$ multiplied by the probability that the sequential process stopped at $k$. Thus, the probability of category 5 is the probability that the sequential process did not fall in 0, 1, 2, 3, or 4 multiplied by the probability that the sequential process stopped at 5. This makes sense why the probability of category 36 is 1. There is no category after 36, so once you multiply all of the previous probabilities with the current category, you get 1. This is the reason for the "cumulative-like" shape of the plot. But if the coefficients were truly cumulative, the probability could not decreases as $k$ increases. 

### Posterior predictive samples

Again, using the posterior predictive samples, we can visualize the model fit against the observed data. In the case of the sequential model, the model does an alright job of capturing the observed frequencies of the categories. For pedagogical purposes, this fit is sufficient.


```python
idata_pps = model.predict(idata=idata, kind="pps", inplace=False)

bins = np.arange(35)
fig, ax = plt.subplots(figsize=(7, 3))
ax = plot_ppc_discrete(idata_pps, bins, ax)
ax.set_xlabel("Response category")
ax.set_ylabel("Count")
ax.set_title("Sequential model - Posterior Predictive Distribution");
```


    
![png](2023-09-29-ordinal-models-bambi_files/2023-09-29-ordinal-models-bambi_45_0.png)
    


## Summary

This notebook demonstrated how to fit cumulative and sequential ordinal regression models using Bambi. Cumulative models focus on modeling the cumulative probabilities of an ordinal outcome variable taking on values up to and including a certain category, whereas a sequential model focuses on modeling the probability that an ordinal outcome variable stops at a particular category, rather than continuing to higher categories. To achieve this, both models assume that the reponse variable originates from a categorization of a latent continuous variable $Z$. However, the cumulative model assumes that there are $K$ thresholds $\tau_k$ that partition $Z$ into $K+1$ observable, ordered categories of $Y$. The sequential model assumes that for every category $k$ there is a latent continuous variable $Z$ that determines the transition between categories $k$ and $k+1$; thus, a threshold $\tau$ belongs to each latent process.

Cumulative models can be used in situations where the outcome variable is on the Likert scale, and you are interested in understanding the impact of predictors on the probability of reaching or exceeding specific categories. Sequential models are particularly useful when you are interested in understanding the predictors that influence the decision to stop at a specific response level. It's well-suited for analyzing data where categories represent stages, and the focus is on the transitions between these stages.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Fri Sep 15 2023
    
    Python implementation: CPython
    Python version       : 3.11.0
    IPython version      : 8.13.2
    
    bambi     : 0.13.0.dev0
    arviz     : 0.15.1
    numpy     : 1.24.2
    pandas    : 2.0.1
    matplotlib: 3.7.1
    
    Watermark: 2.3.1
    

