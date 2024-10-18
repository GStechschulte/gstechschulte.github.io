---
title: 'Google Summer of Code - Average Predictive Comparisons'
date: 2023-06-30
author: 'Gabriel Stechschulte'
draft: false
math: true
categories: ['probabilistic-programming', 'gsoc']
---

<!--eofm-->

It is currently the end of week five of Google Summer of Code 2023. According to the original deliverables table outlined in my proposal, the goal was to have opened a draft PR for the core functionality of the `plot_comparisons`. Subsequently, week six and seven were to be spent further developing the `plot_comparisons` function, and writing tests and a demo notebook for the documentation, respectively. However, at the end of week five, I have a PR open with the majority of the functionality that [marginaleffects](https://vincentarelbundock.github.io/marginaleffects/) has. In addition, I also exposed the `comparisons` function, added tests (which can and will be improved), and have started on documentation.

This is a test $x + y = z$

# Plot Comparisons

`comparisons` and `plot_comparisons` are a part of Bambi's sub-package `interpret` that feature a set of functions used to interpret complex regression models. This sub-package is inspired by the R package [marginaleffects](https://vincentarelbundock.github.io/marginaleffects/articles/predictions.html#conditional-adjusted-predictions-plot). These two functions allow the modeler to **compare** the predictions made by a model for different contrast and covariate values. Below, it is described why comparing predictions is useful in interpreting generalized linear models (GLMs), how this methodology is implemented in Bambi, and how to use `comparisons` and `plot_comparisons`. It is assumed that the reader is familiar with the basics of GLMs. If not, refer to the Bambi [Basic Building Blocks](https://bambinos.github.io/bambi/notebooks/how_bambi_works.html#Link-functions) example.

Due to the link function in a GLM, there are typically three quantities of interest to interpret:

1. the linear predictor $\eta = X\beta$ where $X$ is an $n$ x $p$ matrix of explanatory variables.
2. the mean $\mu = g^{-1}(\eta)$ where the link function $g(\cdot)$ relates the linear predictor to the mean of the outcome variable $\mu = g^{-1}(\eta) = g^{-1}(X\beta)$
3. the response variable $Y \sim \mathcal{D}(\mu, \theta)$ where $\mu$ is the mean parameter and $\theta$ is (possibly) a vector that contains all the other "auxillary" parameters of the distribution.

Often, with GLMs, $\eta$ is linear in the parameters, but nonlinear in relation of inputs to the outcome $Y$ due to the link function $g$. Thus, as modelers, we are usually more interested in interpreting (2) and (3). For example, in logistic regression, the linear predictor is on the log-odds scale, but the quantity of interest is on the probability scale. In Poisson regression, the linear predictor is on the log-scale, but the response variable is on the count scale. Referring back to logistic regression, a specified difference in one of the $x$ variables does _not_ correspond to a constant difference in the the probability of the outcome.

It is often helpful with GLMs, for the modeler and audience, to have a summary that gives the expected difference in the outcome corresponding to a unit difference in each of the input variables. Thus, the goal of `comparisons` and `plot_comparisons` is to provide the modeler with a summary and visualization of the average predicted difference.

## Average Predictive Differences

Here, we adopt the notation from Chapter 14.4 of [Regression and Other Stories](https://avehtari.github.io/ROS-Examples/) to describe average predictive differences. Assume we have fit a Bambi model predicting an outcome $Y$ based on inputs $X$ and parameters $\theta$. Consider the following scalar inputs:

$$w: \text{the input of interest}$$
$$c: \text{all the other inputs}$$
$$X = (w, c)$$

Suppose for the input of interest, we are interested in comparing $w^{\text{high}}$ to $w^{\text{low}}$ (perhaps age = $60$ and $40$ respectively) with all other inputs $c$ held constant. The _predictive difference_ in the outcome changing **only** $w$ is:

$$\text{average predictive difference} = \mathbb{E}(y|w^{\text{high}}, c, \theta) - \mathbb{E}(y|w^{\text{low}}, c, \theta)$$

Selecting the maximum and minimum values of $w$ and averaging over all other inputs $c$ in the data gives you a new "hypothetical" dataset and corresponds to counting all pairs of transitions of $(w^\text{low})$ to $(w^\text{high})$, i.e., differences in $w$ with $c$ held constant. The difference between these two terms is the average predictive difference.

### Computing Average Predictive Differences

The objective of `comparisons` and `plot_comparisons` is to compute the expected difference in the outcome corresponding to three different scenarios for $w$ and $c$ where $w$ is either provided by the user, else a default value is computed by Bambi (described in the default values section). The three scenarios are:

1. user provided values for $c$.
2. a grid of equally spaced and central values for $c$.
3. empirical distribution (original data used to fit the model) for $c$.

In the case of (1) and (2) above, Bambi assembles all pairwise combinations (transitions) of $w$ and $c$ into a new "hypothetical" dataset. In (3), Bambi uses the original $c$, but replaces $w$ with the user provided value or the default value computed by Bambi. In each scenario, predictions are made on the data using the fitted model. Once the predictions are made, comparisons are computed using the posterior samples by taking the difference in the predicted outcome for each pair of transitions. The average of these differences is the average predictive difference.

Thus, the goal of `comparisons` and `plot_comparisons` is to provide the modeler with a summary and visualization of the average predictive difference. Below, we demonstrate how to compute and plot average predictive differences with `comparisons` and `plot_comparions` using several examples.


```python
import arviz as az
import numpy as np
import pandas as pd

import bambi as bmb
```

## Zero Inflated Poisson

We model and predict how many fish are caught by visitors at a state park using survey [data]("https://stats.idre.ucla.edu/stat/data/fish.csv"). Many visitors catch zero fish, either because they did not fish at all, or because they were unlucky. We would like to explicitly model this bimodal behavior (zero versus non-zero) using a Zero Inflated Poisson model, and to compare how different inputs of interest $w$ and other covariate values $c$ are associated with the number of fish caught. The dataset contains data on 250 groups that went to a state park to fish. Each group was questioned about how many fish they caught (`count`), how many children were in the group (`child`), how many people were in the group (`persons`), if they used a live bait and whether or not they brought a camper to the park (`camper`).


```python
fish_data = pd.read_stata("http://www.stata-press.com/data/r11/fish.dta")
cols = ["count", "livebait", "camper", "persons", "child"]
fish_data = fish_data[cols]
fish_data["livebait"] = pd.Categorical(fish_data["livebait"])
fish_data["camper"] = pd.Categorical(fish_data["camper"])
```


```python
fish_model = bmb.Model(
    "count ~ livebait + camper + persons + child",
    fish_data,
    family='zero_inflated_poisson'
)

fish_idata = fish_model.fit(
    draws=1000,
    target_accept=0.95,
    random_seed=1234,
    chains=4
)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [count_psi, Intercept, livebait, camper, persons, child]




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:03&lt;00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 4 seconds.


### User Provided Values

First, an example of scenario 1 (user provided values) is given below. In both `plot_comparisons` and `comparisons`, $w$ and $c$ are represented by `contrast` and `conditional`, respectively. The modeler has the ability to pass their own values for `contrast` and `conditional` by using a dictionary where the key-value pairs are the covariate and value(s) of interest. For example, if we wanted to compare the number of fish caught for $4$ versus $1$ `persons` conditional on a range of `child` and `livebait` values, we would pass the following dictionary in the code block below. By default, for $w$, Bambi compares $w^\text{high}$ to $w^\text{low}$. Thus, in this example, $w^\text{high}$ = 4 and $w^\text{low}$ = 1. The user is not limited to passing a list for the values. A `np.array` can also be used. Furthermore, Bambi by default, maps the order of the dict keys to the main, group, and panel of the matplotlib figure. Below, since `child` is the first key, this is used for the x-axis, and `livebait` is used for the group (color). If a third key was passed, it would be used for the panel (facet).


```python
fig, ax = bmb.interpret.plot_comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast={"persons": [1, 4]},
    conditional={"child": [0, 1, 2], "livebait": [0, 1]},
)
fig.set_size_inches(7, 3)
```



![png](2023-06-30-gsoc-update-comparisons_files/2023-06-30-gsoc-update-comparisons_11_0.png)



The plot above shows that, comparing $4$ to $1$ persons given $0$ children and using livebait, the expected difference is about $26$ fish. When not using livebait, the expected difference decreases substantially to about $5$ fish. Using livebait with a group of people is associated with a much larger expected difference in the number of fish caught.

`comparisons` can be called to view a summary dataframe that includes the term $w$ and its contrast, the specified `conditional` covariate, and the expected difference in the outcome with the uncertainty interval (by default the 94% highest density interval is computed).


```python
bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast={"persons": [1, 4]},
    conditional={"child": [0, 1, 2], "livebait": [0, 1]},
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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>child</th>
      <th>livebait</th>
      <th>camper</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1.0, 4.0)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.834472</td>
      <td>2.563472</td>
      <td>7.037150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1.0, 4.0)</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.423188</td>
      <td>23.739729</td>
      <td>29.072748</td>
    </tr>
    <tr>
      <th>2</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1.0, 4.0)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.202003</td>
      <td>0.631629</td>
      <td>1.780965</td>
    </tr>
    <tr>
      <th>3</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1.0, 4.0)</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.571943</td>
      <td>5.469275</td>
      <td>7.642248</td>
    </tr>
    <tr>
      <th>4</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1.0, 4.0)</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.301384</td>
      <td>0.143676</td>
      <td>0.467608</td>
    </tr>
    <tr>
      <th>5</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1.0, 4.0)</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.648417</td>
      <td>1.140415</td>
      <td>2.187190</td>
    </tr>
  </tbody>
</table>
</div>



But why is `camper` also in the summary dataframe? This is because in order to peform predictions, Bambi is expecting a value for each covariate used to fit the model. Additionally, with GLM models, average predictive comparisons are conditional in the sense that the estimate depends on the values of all the covariates in the model. Thus, for unspecified covariates, `comparisons` and `plot_comparisons` computes a default value (mean or mode based on the data type of the covariate). Thus, $c$ = `child`, `livebait`, `camper`. Each row in the summary dataframe is read as "comparing $4$ to $1$ persons conditional on $c$, the expected difference in the outcome is $y$."

### Multiple contrast values

Users can also perform comparisons on multiple contrast values. For example, if we wanted to compare the number of fish caught between $(1, 2)$, $(1, 4)$, and $(2, 4)$ `persons` conditional on a range of values for `child` and `livebait`.


```python
multiple_values = bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast={"persons": [1, 2, 4]},
    conditional={"child": [0, 1, 2], "livebait": [0, 1]}
)

multiple_values
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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>child</th>
      <th>livebait</th>
      <th>camper</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 2)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.527627</td>
      <td>0.295451</td>
      <td>0.775465</td>
    </tr>
    <tr>
      <th>1</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 2)</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.883694</td>
      <td>2.605690</td>
      <td>3.177685</td>
    </tr>
    <tr>
      <th>2</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 2)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.131319</td>
      <td>0.067339</td>
      <td>0.195132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 2)</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.717965</td>
      <td>0.592968</td>
      <td>0.857893</td>
    </tr>
    <tr>
      <th>4</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 2)</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.032960</td>
      <td>0.015212</td>
      <td>0.052075</td>
    </tr>
    <tr>
      <th>5</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 2)</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.180270</td>
      <td>0.123173</td>
      <td>0.244695</td>
    </tr>
    <tr>
      <th>6</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 4)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.834472</td>
      <td>2.563472</td>
      <td>7.037150</td>
    </tr>
    <tr>
      <th>7</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 4)</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.423188</td>
      <td>23.739729</td>
      <td>29.072748</td>
    </tr>
    <tr>
      <th>8</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 4)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.202003</td>
      <td>0.631629</td>
      <td>1.780965</td>
    </tr>
    <tr>
      <th>9</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 4)</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.571943</td>
      <td>5.469275</td>
      <td>7.642248</td>
    </tr>
    <tr>
      <th>10</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 4)</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.301384</td>
      <td>0.143676</td>
      <td>0.467608</td>
    </tr>
    <tr>
      <th>11</th>
      <td>persons</td>
      <td>diff</td>
      <td>(1, 4)</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.648417</td>
      <td>1.140415</td>
      <td>2.187190</td>
    </tr>
    <tr>
      <th>12</th>
      <td>persons</td>
      <td>diff</td>
      <td>(2, 4)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.306845</td>
      <td>2.267097</td>
      <td>6.280005</td>
    </tr>
    <tr>
      <th>13</th>
      <td>persons</td>
      <td>diff</td>
      <td>(2, 4)</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>23.539494</td>
      <td>20.990931</td>
      <td>26.240169</td>
    </tr>
    <tr>
      <th>14</th>
      <td>persons</td>
      <td>diff</td>
      <td>(2, 4)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.070683</td>
      <td>0.565931</td>
      <td>1.585718</td>
    </tr>
    <tr>
      <th>15</th>
      <td>persons</td>
      <td>diff</td>
      <td>(2, 4)</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.853978</td>
      <td>4.858957</td>
      <td>6.848519</td>
    </tr>
    <tr>
      <th>16</th>
      <td>persons</td>
      <td>diff</td>
      <td>(2, 4)</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.268423</td>
      <td>0.124033</td>
      <td>0.412274</td>
    </tr>
    <tr>
      <th>17</th>
      <td>persons</td>
      <td>diff</td>
      <td>(2, 4)</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.468147</td>
      <td>1.024800</td>
      <td>1.960934</td>
    </tr>
  </tbody>
</table>
</div>



Notice how the contrast $w$ varies while the covariates $c$ are held constant. Currently, however, plotting multiple contrast values can be difficult to interpret since the contrast is "abstracted" away onto the y-axis. Thus, it would be difficult to interpret which portion of the plot corresponds to which contrast value. Therefore, it is currently recommended that if you want to plot multiple contrast values, call `comparisons` directly to obtain the summary dataframe and plot the results yourself.

### Default contrast and conditional values

Now, we move onto scenario 2 described above (grid of equally spaced and central values) in computing average predictive comparisons. You are not required to pass values for `contrast` and `conditional`. If you do not pass values, Bambi will compute default values for you. Below, it is described how these default values are computed.

The default value for `contrast` is a _centered difference_ at the mean for a contrast variable with a numeric dtype, and _unique levels_ for a contrast varaible with a categorical dtype. For example, if the modeler is interested in the comparison of a $5$ unit increase in $w$ where $w$ is a numeric variable, Bambi computes the mean and then subtracts and adds $2.5$ units to the mean to obtain a _centered difference_. By default, if no value is passed for the contrast covariate, Bambi computes a one unit centered difference at the mean. For example, if only `contrast="persons"` is passed, then $\pm$ $0.5$ is applied to the mean of persons. If $w$ is a categorical variable, Bambi computes and returns the unique levels. For example, if $w$ has levels ["high scool", "vocational", "university"], Bambi computes and returns the unique values of this variable.

The default values for `conditional` are more involved. Currently, by default, if a dict or list is passed to `conditional`, Bambi uses the ordering (keys if dict and elements if list) to determine which covariate to use as the main, group (color), and panel (facet) variable. This is the same logic used in `plot_comparisons` described above. Subsequently, the default values used for the `conditional` covariates depend on their ordering **and** dtype. Below, the psuedocode used for computing default values covariates passed to `conditional` is outlined:

```python
if v == "main":

    if v == numeric:
        return np.linspace(v.min(), v.max(), 50)
    elif v == categorical:
        return np.unique(v)

elif v == "group":

    if v == numeric:
        return np.quantile(v, np.linspace(0, 1, 5))
    elif v == categorical:
        return np.unique(v)

elif v == "panel":

    if v == numeric:
        return np.quantile(v, np.linspace(0, 1, 5))
    elif v == categorical:
        return np.unique(v)
```

Thus, letting Bambi compute default values for `conditional` is equivalent to creating a hypothetical "data grid" of new values. Lets say we are interested in comparing the number of fish caught for the contrast `livebait` conditional on `persons` and `child`. This time, lets call `comparisons` first to gain an understanding of the data generating the plot.


```python
contrast_df = bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=["persons", "child"],
)

contrast_df.head(10)
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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>persons</th>
      <th>child</th>
      <th>camper</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.694646</td>
      <td>1.252803</td>
      <td>2.081207</td>
    </tr>
    <tr>
      <th>1</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.422448</td>
      <td>0.299052</td>
      <td>0.551766</td>
    </tr>
    <tr>
      <th>2</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.026923</td>
      <td>0.012752</td>
      <td>0.043035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.061224</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.787412</td>
      <td>1.342979</td>
      <td>2.203158</td>
    </tr>
    <tr>
      <th>4</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.061224</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.445555</td>
      <td>0.317253</td>
      <td>0.580117</td>
    </tr>
    <tr>
      <th>5</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.061224</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.028393</td>
      <td>0.013452</td>
      <td>0.045276</td>
    </tr>
    <tr>
      <th>6</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.122449</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.885270</td>
      <td>1.422937</td>
      <td>2.313218</td>
    </tr>
    <tr>
      <th>7</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.122449</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.469929</td>
      <td>0.335373</td>
      <td>0.609249</td>
    </tr>
    <tr>
      <th>8</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.122449</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.029944</td>
      <td>0.014165</td>
      <td>0.047593</td>
    </tr>
    <tr>
      <th>9</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.183674</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.988500</td>
      <td>1.501650</td>
      <td>2.424762</td>
    </tr>
  </tbody>
</table>
</div>



As `livebait` was encoded as a categorical dtype, Bambi returned the unique levels of $[0, 1]$ for the contrast. `persons` and `child` were passed as the first and second element and thus act as the main and group variables, respectively. It can be see from the output above, that an equally spaced grid was used to compute the values for `persons`, whereas a quantile based grid was used for `child`. Furthermore, as `camper` was unspecified, the mode was used as the default value. Lets go ahead and plot the commparisons.


```python
fig, ax = bmb.interpret.plot_comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=["persons", "child"],
)
fig.set_size_inches(7, 3)
```



![png](2023-06-30-gsoc-update-comparisons_files/2023-06-30-gsoc-update-comparisons_21_0.png)



The plot shows us that the expected differences in fish caught comparing a group of people who use livebait and no livebait is not only conditional on the number of persons, but also children. However, the plotted comparisons for `child` = $3$ is difficult to interpret on a single plot. Thus, it can be useful to pass specific `group` and `panel` arguments to aid in the interpretation of the plot. Therefore, `subplot_kwargs` allows the user to manipulate the plotting by passing a dictionary where the keys are `{"main": ..., "group": ..., "panel": ...}` and the values are the names of the covariates to be plotted. Below, we plot the same comparisons as above, but this time we specify `group` and `panel` to both be `child`.


```python
fig, ax = bmb.interpret.plot_comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=["persons", "child"],
    subplot_kwargs={"main": "persons", "group": "child", "panel": "child"},
    fig_kwargs={"figsize":(12, 3), "sharey": True},
    legend=False
)
```



![png](2023-06-30-gsoc-update-comparisons_files/2023-06-30-gsoc-update-comparisons_23_0.png)



### Unit level contrasts

Evaluating average predictive comparisons at central values for the conditional covariates $c$ can be problematic when the inputs have a large variance since no single central value (mean, median, etc.) is representative of the covariate. This is especially true when $c$ exhibits bi or multimodality. Thus, it may be desireable to use the empirical distribution of $c$ to compute the predictive comparisons, and then average over a specific or set of covariates to obtain the average predictive comparisons. To achieve unit level contrasts, do not pass a parameter into `conditional` and or specify `None`.


```python
unit_level = bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=None,
)

# empirical distribution
print(unit_level.shape[0] == fish_model.data.shape[0])
unit_level.head(10)
```

    True





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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>camper</th>
      <th>child</th>
      <th>persons</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.864408</td>
      <td>0.627063</td>
      <td>1.116105</td>
    </tr>
    <tr>
      <th>1</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.694646</td>
      <td>1.252803</td>
      <td>2.081207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.864408</td>
      <td>0.627063</td>
      <td>1.116105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.009094</td>
      <td>0.755449</td>
      <td>1.249551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.864408</td>
      <td>0.627063</td>
      <td>1.116105</td>
    </tr>
    <tr>
      <th>5</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.453235</td>
      <td>0.964674</td>
      <td>1.956434</td>
    </tr>
    <tr>
      <th>6</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.233247</td>
      <td>0.900295</td>
      <td>1.569891</td>
    </tr>
    <tr>
      <th>7</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.188019</td>
      <td>0.090328</td>
      <td>0.289560</td>
    </tr>
    <tr>
      <th>8</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.606361</td>
      <td>0.390571</td>
      <td>0.818549</td>
    </tr>
    <tr>
      <th>9</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.694646</td>
      <td>1.252803</td>
      <td>2.081207</td>
    </tr>
  </tbody>
</table>
</div>




```python
# empirical (observed) data used to fit the model
fish_model.data.head(10)
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
      <th>count</th>
      <th>livebait</th>
      <th>camper</th>
      <th>persons</th>
      <th>child</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Above, `unit_level` is the comparisons summary dataframe and `fish_model.data` is the empirical data. Notice how the values for $c$ are identical in both dataframes. However, for $w$, the values are different. However, these unit level contrasts are difficult to interpret as each row corresponds to _that_ unit's contrast. Therefore, it is useful to average over (marginalize) the estimates to summarize the unit level predictive comparisons.

#### Marginalizing over covariates

Since the empirical distrubution is used for computing the average predictive comparisons, the same number of rows (250) is returned as the data used to fit the model. To average over a covariate, use the `average_by` argument. If `True` is passed, then `comparisons` averages over all covariates. Else, if a single or list of covariates are passed, then `comparisons` averages by the covariates passed.


```python
# marginalize over all covariates
bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=None,
    average_by=True
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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>3.649691</td>
      <td>2.956185</td>
      <td>4.333621</td>
    </tr>
  </tbody>
</table>
</div>



Passing `True` to `average_by` averages over all covariates and is equivalent to taking the mean of the `estimate` and uncertainty columns. For example:


```python
unit_level = bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=None,
)

unit_level[["estimate", "lower_3.0%", "upper_97.0%"]].mean()
```




    estimate       3.649691
    lower_3.0%     2.956185
    upper_97.0%    4.333621
    dtype: float64



#### Average by subgroups

Averaging over all covariates may not be desired, and you would rather average by a group or specific covariate. To perform averaging by subgroups, users can pass a single or list of covariates to `average_by` to average over specific covariates. For example, if we wanted to average by `persons`:


```python
# average by number of persons
bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=None,
    average_by="persons"
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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>persons</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>1.374203</td>
      <td>1.011290</td>
      <td>1.708711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>2.0</td>
      <td>1.963362</td>
      <td>1.543330</td>
      <td>2.376636</td>
    </tr>
    <tr>
      <th>2</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>3.0</td>
      <td>3.701510</td>
      <td>3.056586</td>
      <td>4.357385</td>
    </tr>
    <tr>
      <th>3</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>4.0</td>
      <td>7.358662</td>
      <td>6.047642</td>
      <td>8.655654</td>
    </tr>
  </tbody>
</table>
</div>




```python
# average by number of persons and camper by passing a list
bmb.interpret.comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=None,
    average_by=["persons", "camper"]
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
      <th>term</th>
      <th>estimate_type</th>
      <th>value</th>
      <th>persons</th>
      <th>camper</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.864408</td>
      <td>0.627063</td>
      <td>1.116105</td>
    </tr>
    <tr>
      <th>1</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.694646</td>
      <td>1.252803</td>
      <td>2.081207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.424598</td>
      <td>1.078389</td>
      <td>1.777154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.344439</td>
      <td>1.872191</td>
      <td>2.800661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.429459</td>
      <td>1.871578</td>
      <td>2.964242</td>
    </tr>
    <tr>
      <th>5</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4.443540</td>
      <td>3.747840</td>
      <td>5.170052</td>
    </tr>
    <tr>
      <th>6</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.541921</td>
      <td>2.686445</td>
      <td>4.391176</td>
    </tr>
    <tr>
      <th>7</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0.0, 1.0)</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>10.739204</td>
      <td>9.024702</td>
      <td>12.432764</td>
    </tr>
  </tbody>
</table>
</div>



It is still possible to use `plot_comparisons` when passing an argument to `average_by`. In the plot below, the empirical distribution is used to compute unit level contrasts for `livebait` and then averaged over `persons` to obtain the average predictive comparisons. The plot below is similar to the second plot in this notebook. The differences being that: (1) a pairwise transition grid is defined for the second plot above, whereas the empirical distribution is used in the plot below, and (2) in the plot below, we marginalized over the other covariates in the model (thus the reason for not having a `camper` or `child` group and panel, and a reduction in the uncertainty interval).


```python
fig, ax = bmb.interpret.plot_comparisons(
    model=fish_model,
    idata=fish_idata,
    contrast="livebait",
    conditional=None,
    average_by="persons"
)
fig.set_size_inches(7, 3)
```



![png](2023-06-30-gsoc-update-comparisons_files/2023-06-30-gsoc-update-comparisons_37_0.png)



## Logistic Regression

To showcase an additional functionality of `comparisons` and `plot_comparisons`, we fit a logistic regression model to the [titanic dataset](https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv) with interaction terms to model the probability of survival. The titanic dataset gives the values of four categorical attributes for each of the 2201 people on board the Titanic when it struck an iceberg and sank. The attributes are social class (first class, second class, third class, crewmember), age, sex (0 = female, 1 = male), and whether or not the person survived (0 = deceased, 1 = survived).


```python
dat = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv", index_col=0)

dat["PClass"] = dat["PClass"].str.replace("[st, nd, rd]", "", regex=True)
dat["PClass"] = dat["PClass"].str.replace("*", "0").astype(int)
dat["PClass"] = dat["PClass"].replace(0, np.nan)
dat["PClass"] = pd.Categorical(dat["PClass"], ordered=True)
dat["SexCode"] = pd.Categorical(dat["SexCode"], ordered=True)

dat = dat.dropna(axis=0, how="any")
```


```python
titanic_model = bmb.Model(
    "Survived ~ PClass * SexCode * Age",
    data=dat,
    family="bernoulli"
)
titanic_idata = titanic_model.fit(draws=1000, target_accept=0.95, random_seed=1234)
```

    Modeling the probability that Survived==1
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [Intercept, PClass, SexCode, PClass:SexCode, Age, PClass:Age, SexCode:Age, PClass:SexCode:Age]




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:15&lt;00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 16 seconds.


### Comparison types

`comparisons` and `plot_comparisons` also allow you to specify the type of comparison to be computed. By default, a difference is used. However, it is also possible to take the ratio where comparisons would then become _average predictive ratios_. To achieve this, pass `"ratio"` into the argument `comparison_type`. Using different comparison types offers a way to produce alternative insights; especially when there are interaction terms as the value of one covariate depends on the value of the other covariate.


```python
fig, ax = bmb.interpret.plot_comparisons(
    model=titanic_model,
    idata=titanic_idata,
    contrast={"PClass": [1, 3]},
    conditional=["Age", "SexCode"],
    comparison_type="ratio",
    subplot_kwargs={"main": "Age", "group": "SexCode", "panel": "SexCode"},
    fig_kwargs={"figsize":(12, 3), "sharey": True},
    legend=False

)
```



![png](2023-06-30-gsoc-update-comparisons_files/2023-06-30-gsoc-update-comparisons_42_0.png)



The left panel shows that the ratio of the probability of survival comparing `PClass` $3$ to $1$ conditional on `Age` is non-constant. Whereas the right panel shows an approximately constant ratio in the probability of survival comparing `PClass` $3$ to $1$ conditional on `Age`.

### Conclusion

Overall, solid progress has been made with `comparisons` and `plot_comparisons` to give the Bambi modeller insights into GLMs. One of the most difficult aspects to program is the building of the contrast dataframe (the output above) as it requires tedious shape handling to ensure the predictive comparisons are "mapped" to the correct contrast and conditional values. Writing better tests to ensure shapes are correct is something I am also working on. Additionally, there are still "nice to have" features such as:

* cross-contrasts
* comparisons other than the predictive difference or ratios (e.g., adjusted risk ratio)
