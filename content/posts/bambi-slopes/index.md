---
title: 'Google Summer of Code - Average Predictive Slopes'
date: 2023-08-01
author: 'Gabriel Stechschulte'
draft: false
categories: ['probabilistic-programming']
---

<!--eofm-->

It is currently the beginning of week ten of Google Summer of Code 2023. According to the original deliverables table outlined in my proposal, the goal was to have opened a draft PR for the basic functionality of the `plot_slopes`. Subsequently, week 11 was reserved to further develop the `plot_slopes` function, and to write tests and a notebook for the documentation, respectively. 

However, at the beginning of week ten, I have a [PR](https://github.com/bambinos/bambi/pull/699) open with the majority of the functionality that [marginaleffects](https://vincentarelbundock.github.io/marginaleffects/) has for `slopes`. In addition, I also exposed the `slopes` function, added tests, and have a [PR](https://github.com/bambinos/bambi/pull/701) open for the documentation.

Below is the documentation for `slopes` and `plot_slopes` in Bambi, and is a culmination of the work completed in weeks five through nine.

# Plot Slopes

Bambi's sub-package `interpret` features a set of functions to help interpret complex regression models. The sub-package is inspired by the R package [marginaleffects](https://vincentarelbundock.github.io/marginaleffects/articles/predictions.html#conditional-adjusted-predictions-plot). In this notebook we will discuss two functions `slopes` and `plot_slopes`. These two functions allow the modeler to easier interpret slopes, either by a inspecting a summary output or plotting them.

Below, it is described why estimating the slope of the prediction function is useful in interpreting generalized linear models (GLMs), how this methodology is implemented in Bambi, and how to use `slopes` and `plot_slopes`. It is assumed that the reader is familiar with the basics of GLMs. If not, refer to the Bambi [Basic Building Blocks](https://bambinos.github.io/bambi/notebooks/how_bambi_works.html#Link-functions) example.

## Interpretation of Regression Coefficients

Assuming we have fit a linear regression model of the form

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k + \epsilon$$

the "safest" interpretation of the regression coefficients $\beta$ is as a comparison between two groups of items that differ by $1$ in the relevant predictor variable $x_i$ while being identical in all the other predictors. Formally, the predicted difference between two items $i$ and $j$ that differ by an amount $n$ on predictor $k$, but are identical on all other predictors, the predicted difference is $y_i - y_j$ is $\beta_kx$, on average.

However, once we move away from a regression model with a Gaussian response, the identity function, and no interaction terms, the interpretation of the coefficients are not as straightforward. For example, in a logistic regression model, the coefficients are on a different scale and are measured in logits (log odds), not probabilities or percentage points. Thus, you cannot interpret the coefficents as a "one unit increase in $x_k$ is associated with an $n$ percentage point decrease in $y$". First, the logits must be converted to the probability scale. Secondly, a one unit change in $x_k$ may produce a larger or smaller change in the outcome, depending upon how far away from zero the logits are. 

`slopes` and `plot_slopes`, by default, computes quantities of interest on the response scale for GLMs. For example, for a logistic regression model, this is the probability scale, and for a Poisson regression model, this is the count scale.

### Interpreting interaction effects

Specifying interactions in a regression model is a way of allowing parameters to be conditional on certain aspects of the data. By contrast, for a model with no interactions, the parameters are **not** conditional and thus, the value of one parameter is not dependent on the value of another covariate. However, once interactions exist, multiple parameters are always in play at the same time. Additionally, interactions can be specified for either categorical, continuous, or both types of covariates. Thus, making the interpretation of the parameters more difficult.

With GLMs, every covariate essentially interacts with itself because of the link function. To demonstrate parameters interacting with themselves, consider the mean of a Gaussian linear model with an identity link function

$$\mu = \alpha + \beta x$$

where the rate of change in $\mu$ with respect to $x$ is just $\beta$, i.e., the rate of change is constant no matter what the value of $x$ is. But when we consider GLMs with link functions used to map outputs to exponential family distribution parameters, calculating the derivative of the mean output $\mu$ with respect to the predictor is not as straightforward as in the Gaussian linear model. For example, computing the rate of change in a binomial probability $p$ with respect to $x$

$$p = \frac{exp(\alpha + \beta x)}{1 + exp(\alpha + \beta x)}$$

And taking the derivative of $p$ with respect to $x$ yields

$$\frac{\partial p}{\partial x} = \frac{\beta}{2(1 + cosh(\alpha + \beta x))}$$

Since $x$ appears in the derivative, the impact of a change in $x$ depends upon $x$, i.e., an interaction with itself even though no interaction term was specified in the model.Thus, visualizing the rate of change in the mean response with respect to a covariate $x$ becomes a useful tool in interpreting GLMs.

## Average Predictive Slopes

Here, we adopt the notation from Chapter 14.4 of [Regression and Other Stories](https://avehtari.github.io/ROS-Examples/) to first describe average predictive differences which is essential to computing `slopes`, and then secondly, average predictive slopes. Assume we have fit a Bambi model predicting an outcome $Y$ based on inputs $X$ and parameters $\theta$. Consider the following scalar inputs:

$$w: \text{the input of interest}$$
$$c: \text{all the other inputs}$$
$$X = (w, c)$$

In contrast to `comparisons`, for `slopes` we are interested in comparing $w^{\text{value}}$ to $w^{\text{value}+\epsilon}$ (perhaps age = 60 and 60.0001 respectively) with all other inputs $c$ held constant. The _predictive difference_ in the outcome changing **only** $w$ is:

$$\text{average predictive difference} = \mathbb{E}(y|w^{\text{value}}, c, \theta) - \mathbb{E}(y|w^{\text{value}+\epsilon}, c, \theta)$$

Selecting $w$ and $w^{\text{value}+\epsilon}$ and averaging over all other inputs $c$ in the data gives you a new "hypothetical" dataset and corresponds to counting all pairs of transitions of $(w^\text{value})$ to $(w^{\text{value}+\epsilon})$, i.e., differences in $w$ with $c$ held constant. The difference between these two terms is the average predictive difference.

However, to obtain the slope estimate, we need to take the above formula and divide by $\epsilon$ to obtain the _average predictive slope_:

$$\text{average predictive slope} = \frac{\mathbb{E}(y|w^{\text{value}}, c, \theta) - \mathbb{E}(y|w^{\text{value}+\epsilon}, c, \theta)}{\epsilon}$$

## Computing Slopes

The objective of `slopes` and `plot_slopes` is to compute the rate of change (slope) in the mean of the response $y$ with respect to a small change $\epsilon$ in the predictor $x$ conditional on other covariates $c$ specified in the model. $w$ is specified by the user and the original value is either provided by the user, else a default value (the mean) is computed by Bambi. The values for the other covariates $c$ specified in the model can be determined under the following three scenarios:

1. user provided values 
2. a grid of equally spaced and central values
3. empirical distribution (original data used to fit the model)

In the case of (1) and (2) above, Bambi assembles all pairwise combinations (transitions) of $w$ and $c$ into a new "hypothetical" dataset. In (3), Bambi uses the original $c$, and adds a small amount $\epsilon$ to each unit of observation's $w$. In each scenario, predictions are made on the data using the fitted model. Once the predictions are made, comparisons are computed using the posterior samples by taking the difference in the predicted outcome for each pair of transitions and dividing by $\epsilon$. The average of these slopes is the average predictive slopes.

For variables $w$ with a string or categorical data type, the `comparisons` function is called to compute the expected difference in group means. Please refer to the [comparisons](https://bambinos.github.io/bambi/notebooks/plot_comparisons.html) documentation for more details.

Below, we present several examples showing how to use Bambi to perform these computations for us, and to return either a summary dataframe, or a visualization of the results.


```python
import arviz as az
import pandas as pd

import bambi as bmb
```

## Logistic Regression

To demonstrate `slopes` and `plot_slopes`, we will use the [well switching dataset](https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wells.html) to model the probability a household in Bangladesh switches water wells. The data are for an area of Arahazar Upazila, Bangladesh. The researchers labelled each well with its level of arsenic and an indication of whether the well was “safe” or “unsafe”. Those using unsafe wells were encouraged to switch. After several years, it was determined whether each household using an unsafe well had changed its well. The data contains $3020$ observations on the following five variables:

- `switch`: a factor with levels `no` and `yes` indicating whether the household switched to a new well
- `arsenic`: the level of arsenic in the old well (measured in micrograms per liter)
- `dist`: the distance to the nearest safe well (measured in meters)
- `assoc`: a factor with levels `no` and `yes` indicating whether the household is a member of an arsenic education group
- `educ`: years of education of the household head

First, a logistic regression model with no interactions is fit to the data. Subsequently, to demonstrate the benefits of `plot_slopes` in interpreting interactions, we will fit a logistic regression model with an interaction term.


```python
data = pd.read_csv("http://www.stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat", sep=" ")
data["switch"] = pd.Categorical(data["switch"])
data["dist100"] = data["dist"] / 100
data["educ4"] = data["educ"] / 4
data.head()
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
      <th>switch</th>
      <th>arsenic</th>
      <th>dist</th>
      <th>assoc</th>
      <th>educ</th>
      <th>dist100</th>
      <th>educ4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.36</td>
      <td>16.826000</td>
      <td>0</td>
      <td>0</td>
      <td>0.16826</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.71</td>
      <td>47.321999</td>
      <td>0</td>
      <td>0</td>
      <td>0.47322</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2.07</td>
      <td>20.966999</td>
      <td>0</td>
      <td>10</td>
      <td>0.20967</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1.15</td>
      <td>21.486000</td>
      <td>0</td>
      <td>12</td>
      <td>0.21486</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1.10</td>
      <td>40.874001</td>
      <td>1</td>
      <td>14</td>
      <td>0.40874</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
well_model = bmb.Model(
    "switch ~ dist100 + arsenic + educ4",
    data,
    family="bernoulli"
)

well_idata = well_model.fit(
    draws=1000, 
    target_accept=0.95, 
    random_seed=1234, 
    chains=4
)
```

### User provided values

First, an example of scenario 1 (user provided values) is given below. In both `plot_slopes` and `slopes`, $w$ and $c$ are represented by `wrt` (with respect to) and `conditional`, respectively. The modeler has the ability to pass their own values for `wrt` and `conditional` by using a dictionary where the key-value pairs are the covariate and value(s) of interest.

For example, if we wanted to compute the slope of the probability of switching wells for a typical `arsenic` value of $1.3$ conditional on a range of `dist` and `educ` values, we would pass the following dictionary in the code block below. By default, for $w$, Bambi compares $w^\text{value}$ to $w^{\text{value} + \epsilon}$ where $\epsilon =$ `1e-4`. However, the value for $\epsilon$ can be changed by passing a value to the argument `eps`. 

Thus, in this example, $w^\text{value} = 1.3$ and $w^{\text{value} + \epsilon} = 1.3001$. The user is not limited to passing a list for the values. A `np.array` can also be used. Furthermore, Bambi by default, maps the order of the dict keys to the main, group, and panel of the matplotlib figure. Below, since `dist100` is the first key, this is used for the x-axis, and `educ4` is used for the group (color). If a third key was passed, it would be used for the panel (facet).


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model,
    well_idata,
    wrt={"arsenic": 1.3},
    conditional={"dist100": [0.20, 0.50, 0.80], "educ4": [1.00, 1.20, 2.00]},
)
fig.set_size_inches(7, 3)
fig.axes[0].set_ylabel("Slope of Well Switching Probability");
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_12_0.png)
    


The plot above shows that, for example, conditional on `dist100` $= 0.2$ and `educ4` $= 1.0$ a unit increase in `arsenic` is associated with households being $11$% less likely to switch wells. Notice that even though we fit a logistic regression model where the coefficients are on the log-odds scale, the `slopes` function returns the slope on the probability scale. Thus, we can interpret the y-axis (slope) as the expected change in the probability of switching wells for a unit increase in `arsenic` conditional on the specified covariates.

`slopes` can be called directly to view a summary dataframe that includes the term name, estimate type (discussed in detail in the _interpreting coefficients as an elasticity_ section), values $w$ used to compute the estimate, the specified conditional covariates $c$, and the expected slope of the outcome with the uncertainty interval (by default the $94$% highest density interval is computed).


```python
bmb.interpret.slopes(
    well_model,
    well_idata,
    wrt={"arsenic": 1.5},
    conditional={
        "dist100": [0.20, 0.50, 0.80], 
        "educ4": [1.00, 1.20, 2.00]
        }
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
      <th>dist100</th>
      <th>educ4</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>-0.110797</td>
      <td>-0.128775</td>
      <td>-0.092806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>-0.109867</td>
      <td>-0.126725</td>
      <td>-0.091065</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>-0.105618</td>
      <td>-0.122685</td>
      <td>-0.088383</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>-0.116087</td>
      <td>-0.134965</td>
      <td>-0.096843</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.5</td>
      <td>1.2</td>
      <td>-0.115632</td>
      <td>-0.134562</td>
      <td>-0.096543</td>
    </tr>
    <tr>
      <th>5</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>-0.113140</td>
      <td>-0.130448</td>
      <td>-0.093209</td>
    </tr>
    <tr>
      <th>6</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>-0.117262</td>
      <td>-0.136850</td>
      <td>-0.098549</td>
    </tr>
    <tr>
      <th>7</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.8</td>
      <td>1.2</td>
      <td>-0.117347</td>
      <td>-0.136475</td>
      <td>-0.098044</td>
    </tr>
    <tr>
      <th>8</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.8</td>
      <td>2.0</td>
      <td>-0.116957</td>
      <td>-0.135079</td>
      <td>-0.096476</td>
    </tr>
  </tbody>
</table>
</div>



Since all covariates used to fit the model were also specified to compute the slopes, no default value is used for unspecified covariates. A default value is computed for the unspecified covariates because in order to peform predictions, Bambi is expecting a value for each covariate used to fit the model. Additionally, with GLM models, average predictive slopes are conditional in the sense that the estimate depends on the values of **all** the covariates in the model. Thus, for unspecified covariates, `slopes` and `plot_slopes` computes a default value (mean or mode based on the data type of the covariate). Each row in the summary dataframe is read as "the slope (or rate of change) of the probability of switching wells with respect to a small change in $w$ conditional on $c$ is $y$".

### Multiple slope values

Users can also compute slopes on multiple values for `wrt`. For example, if we want to compute the slope of $y$ with respect to `arsenic` $= 1.5$, $2.0$, and $2.5$, simply pass a list or numpy array as the dictionary values for `wrt`. Keeping the conditional covariate and values the same, the following slope estimates are computed below.


```python
multiple_values = bmb.interpret.slopes(
    well_model,
    well_idata,
    wrt={"arsenic": [1.5, 2.0, 2.5]},
    conditional={
        "dist100": [0.20, 0.50, 0.80], 
        "educ4": [1.00, 1.20, 2.00]
        }
)

multiple_values.head(6)
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
      <th>dist100</th>
      <th>educ4</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>-0.110797</td>
      <td>-0.128775</td>
      <td>-0.092806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.0, 2.0001)</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>-0.109867</td>
      <td>-0.126725</td>
      <td>-0.091065</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.5, 2.5001)</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>-0.105618</td>
      <td>-0.122685</td>
      <td>-0.088383</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.5, 1.5001)</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>-0.116087</td>
      <td>-0.134965</td>
      <td>-0.096843</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.0, 2.0001)</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>-0.115632</td>
      <td>-0.134562</td>
      <td>-0.096543</td>
    </tr>
    <tr>
      <th>5</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.5, 2.5001)</td>
      <td>0.2</td>
      <td>1.2</td>
      <td>-0.113140</td>
      <td>-0.130448</td>
      <td>-0.093209</td>
    </tr>
  </tbody>
</table>
</div>



The output above is essentially the same as the summary dataframe when we only passed one value to `wrt`. However, now each element (value) in the list gets a small amount $\epsilon$ added to it, and the slope is calculated for each of these values.

### Conditional slopes

As stated in the _interpreting interaction effects_ section, interpreting coefficients of multiple interaction terms can be difficult and cumbersome. Thus, `plot_slopes` provides an effective way to visualize the conditional slopes of the interaction effects. Below, we will use the same well switching dataset, but with interaction terms. Specifically, one interaction is added between `dist100` and `educ4`, and another between `arsenic` and `educ4`.


```python
well_model_interact = bmb.Model(
    "switch ~ dist100 + arsenic + educ4 + dist100:educ4 + arsenic:educ4",
    data,
    family="bernoulli"
)

well_idata_interact = well_model_interact.fit(
    draws=1000, 
    target_accept=0.95, 
    random_seed=1234, 
    chains=4
)
```


```python
# summary of coefficients
az.summary(well_idata_interact)
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
      <th>Intercept</th>
      <td>-0.097</td>
      <td>0.122</td>
      <td>-0.322</td>
      <td>0.137</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>2259.0</td>
      <td>2203.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>dist100</th>
      <td>1.320</td>
      <td>0.175</td>
      <td>0.982</td>
      <td>1.640</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>2085.0</td>
      <td>2457.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>arsenic</th>
      <td>-0.398</td>
      <td>0.061</td>
      <td>-0.521</td>
      <td>-0.291</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2141.0</td>
      <td>2558.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>educ4</th>
      <td>0.102</td>
      <td>0.080</td>
      <td>-0.053</td>
      <td>0.246</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1935.0</td>
      <td>2184.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>dist100:educ4</th>
      <td>-0.330</td>
      <td>0.106</td>
      <td>-0.528</td>
      <td>-0.136</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2070.0</td>
      <td>2331.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>arsenic:educ4</th>
      <td>-0.079</td>
      <td>0.043</td>
      <td>-0.161</td>
      <td>-0.000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2006.0</td>
      <td>2348.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



The coefficients of the linear model are shown in the table above. The interaction coefficents indicate the slope varies in a continuous fashion with the continuous variable.

A negative value for `arsenic:dist100` indicates that the "effect" of arsenic on the outcome is less negative as distance from the well increases. Similarly, a negative value for `arsenic:educ4` indicates that the "effect" of arsenic on the outcome is more negative as education increases. Remember, these coefficients are still on the logit scale. Furthermore, as more variables and interaction terms are added to the model, interpreting these coefficients becomes more difficult. 

Thus, lets use `plot_slopes` to visually see how the slope changes with respect to `arsenic` conditional on `dist100` and `educ4` changing. Notice in the code block below how parameters are passed to the `subplot_kwargs` and `fig_kwargs` arguments. At times, it can be useful to pass specific `group` and `panel` arguments to aid in the interpretation of the plot. Therefore, `subplot_kwargs` allows the user to manipulate the plotting by passing a dictionary where the keys are `{"main": ..., "group": ..., "panel": ...}` and the values are the names of the covariates to be plotted. `fig_kwargs` are figure level key word arguments such as `figsize` and `sharey`.


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    conditional=["dist100", "educ4"],
    subplot_kwargs={"main": "dist100", "group": "educ4", "panel": "educ4"},
    fig_kwargs={"figsize": (16, 4), "sharey": True},
    legend=False
)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_23_0.png)
    


With interaction terms now defined, it can be seen how the slope of the outcome with respect to `arsenic` differ depending on the value of `educ4`. Especially in the case of `educ4` $= 4.25$, the slope is more "constant", but with greater uncertainty. Lets compare this with the model that does not include any interaction terms.


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model,
    well_idata,
    wrt="arsenic",
    conditional=["dist100", "educ4"],
    subplot_kwargs={"main": "dist100", "group": "educ4", "panel": "educ4"},
    fig_kwargs={"figsize": (16, 4), "sharey": True},
    legend=False
)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_25_0.png)
    


For the non-interaction model, conditional on a range of values for `educ4` and `dist100`, the slopes of the outcome are nearly identical.

### Unit level slopes

Evaluating average predictive slopes at central values for the conditional covariates $c$ can be problematic when the inputs have a large variance since no single central value (mean, median, etc.) is representative of the covariate. This is especially true when $c$ exhibits bi or multimodality. Thus, it may be desireable to use the empirical distribution of $c$ to compute the predictive slopes, and then average over a specific or set of covariates to obtain average slopes. To achieve unit level slopes, do not pass a parameter into `conditional` and or specify `None`.


```python
unit_level = bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    conditional=None
)

# empirical distribution
print(unit_level.shape[0] == well_model_interact.data.shape[0])
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
      <th>dist100</th>
      <th>educ4</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.36, 2.3601)</td>
      <td>0.16826</td>
      <td>0.00</td>
      <td>-0.084280</td>
      <td>-0.105566</td>
      <td>-0.063403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(0.71, 0.7101)</td>
      <td>0.47322</td>
      <td>0.00</td>
      <td>-0.097837</td>
      <td>-0.125057</td>
      <td>-0.070959</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.07, 2.0701)</td>
      <td>0.20967</td>
      <td>2.50</td>
      <td>-0.118093</td>
      <td>-0.139848</td>
      <td>-0.093442</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.15, 1.1501)</td>
      <td>0.21486</td>
      <td>3.00</td>
      <td>-0.150638</td>
      <td>-0.194765</td>
      <td>-0.108946</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(1.1, 1.1001)</td>
      <td>0.40874</td>
      <td>3.50</td>
      <td>-0.161272</td>
      <td>-0.214761</td>
      <td>-0.108663</td>
    </tr>
    <tr>
      <th>5</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(3.9, 3.9001)</td>
      <td>0.69518</td>
      <td>2.25</td>
      <td>-0.073908</td>
      <td>-0.080525</td>
      <td>-0.067493</td>
    </tr>
    <tr>
      <th>6</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.97, 2.9701000000000004)</td>
      <td>0.80711</td>
      <td>1.00</td>
      <td>-0.108482</td>
      <td>-0.123517</td>
      <td>-0.093042</td>
    </tr>
    <tr>
      <th>7</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(3.24, 3.2401000000000004)</td>
      <td>0.55146</td>
      <td>2.50</td>
      <td>-0.088049</td>
      <td>-0.097939</td>
      <td>-0.078020</td>
    </tr>
    <tr>
      <th>8</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(3.28, 3.2801)</td>
      <td>0.52647</td>
      <td>0.00</td>
      <td>-0.087388</td>
      <td>-0.107331</td>
      <td>-0.068076</td>
    </tr>
    <tr>
      <th>9</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>(2.52, 2.5201000000000002)</td>
      <td>0.75072</td>
      <td>0.00</td>
      <td>-0.099035</td>
      <td>-0.129517</td>
      <td>-0.073222</td>
    </tr>
  </tbody>
</table>
</div>




```python
well_model_interact.data.head(10)
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
      <th>switch</th>
      <th>arsenic</th>
      <th>dist</th>
      <th>assoc</th>
      <th>educ</th>
      <th>dist100</th>
      <th>educ4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.36</td>
      <td>16.826000</td>
      <td>0</td>
      <td>0</td>
      <td>0.16826</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.71</td>
      <td>47.321999</td>
      <td>0</td>
      <td>0</td>
      <td>0.47322</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2.07</td>
      <td>20.966999</td>
      <td>0</td>
      <td>10</td>
      <td>0.20967</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1.15</td>
      <td>21.486000</td>
      <td>0</td>
      <td>12</td>
      <td>0.21486</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1.10</td>
      <td>40.874001</td>
      <td>1</td>
      <td>14</td>
      <td>0.40874</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3.90</td>
      <td>69.517998</td>
      <td>1</td>
      <td>9</td>
      <td>0.69518</td>
      <td>2.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2.97</td>
      <td>80.710999</td>
      <td>1</td>
      <td>4</td>
      <td>0.80711</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3.24</td>
      <td>55.146000</td>
      <td>0</td>
      <td>10</td>
      <td>0.55146</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>3.28</td>
      <td>52.646999</td>
      <td>1</td>
      <td>0</td>
      <td>0.52647</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2.52</td>
      <td>75.071999</td>
      <td>1</td>
      <td>0</td>
      <td>0.75072</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Above, `unit_level` is the slopes summary dataframe and `well_model_interact.data` is the empirical data used to fit the model. Notice how the values for $c$ are identical in both dataframes. However, for $w$, the values are the original $w$ value plus $\epsilon$. Thus, the `estimate` value represents the instantaneous rate of change for that unit of observation. However, these unit level slopes are difficult to interpret since each row may have a different slope estimate. Therefore, it is useful to average over (marginalize) the estimates to summarize the unit level predictive slopes.

#### Marginalizing over covariates

Since the empirical distrubution is used for computing the average predictive slopes, the same number of rows ($3020$) is returned as the data used to fit the model. To average over a covariate, use the `average_by` argument. If `True` is passed, then `slopes` averages over all covariates. Else, if a single or list of covariates are passed, then `slopes` averages by the covariates passed.


```python
bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
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
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>-0.111342</td>
      <td>-0.134846</td>
      <td>-0.088171</td>
    </tr>
  </tbody>
</table>
</div>



The code block above is equivalent to taking the mean of the `estimate` and uncertainty columns. For example:


```python
unit_level[["estimate", "lower_3.0%", "upper_97.0%"]].mean()
```




    estimate      -0.111342
    lower_3.0%    -0.134846
    upper_97.0%   -0.088171
    dtype: float64



#### Average by subgroups

Averaging over all covariates may not be desired, and you would rather average by a group or specific covariate. To perform averaging by subgroups, users can pass a single or list of covariates to `average_by` to average over specific covariates. For example, if we wanted to average by `educ4`:


```python
# average by educ4
bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    conditional=None,
    average_by="educ4"
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
      <th>educ4</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.00</td>
      <td>-0.092389</td>
      <td>-0.119320</td>
      <td>-0.068167</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.25</td>
      <td>-0.101704</td>
      <td>-0.126096</td>
      <td>-0.076910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.50</td>
      <td>-0.102112</td>
      <td>-0.122443</td>
      <td>-0.082142</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.75</td>
      <td>-0.106004</td>
      <td>-0.124247</td>
      <td>-0.088132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>1.00</td>
      <td>-0.110580</td>
      <td>-0.127803</td>
      <td>-0.093221</td>
    </tr>
    <tr>
      <th>5</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>1.25</td>
      <td>-0.112334</td>
      <td>-0.128771</td>
      <td>-0.094870</td>
    </tr>
    <tr>
      <th>6</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>1.50</td>
      <td>-0.114875</td>
      <td>-0.132652</td>
      <td>-0.096790</td>
    </tr>
    <tr>
      <th>7</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>1.75</td>
      <td>-0.122557</td>
      <td>-0.142921</td>
      <td>-0.101423</td>
    </tr>
    <tr>
      <th>8</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>2.00</td>
      <td>-0.125187</td>
      <td>-0.148096</td>
      <td>-0.101350</td>
    </tr>
    <tr>
      <th>9</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>2.25</td>
      <td>-0.125367</td>
      <td>-0.150676</td>
      <td>-0.099852</td>
    </tr>
    <tr>
      <th>10</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>2.50</td>
      <td>-0.130748</td>
      <td>-0.159912</td>
      <td>-0.101058</td>
    </tr>
    <tr>
      <th>11</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>2.75</td>
      <td>-0.137422</td>
      <td>-0.170662</td>
      <td>-0.102995</td>
    </tr>
    <tr>
      <th>12</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>3.00</td>
      <td>-0.136103</td>
      <td>-0.172119</td>
      <td>-0.099548</td>
    </tr>
    <tr>
      <th>13</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>3.25</td>
      <td>-0.156941</td>
      <td>-0.202215</td>
      <td>-0.107625</td>
    </tr>
    <tr>
      <th>14</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>3.50</td>
      <td>-0.142571</td>
      <td>-0.186079</td>
      <td>-0.098362</td>
    </tr>
    <tr>
      <th>15</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>3.75</td>
      <td>-0.138336</td>
      <td>-0.181042</td>
      <td>-0.093120</td>
    </tr>
    <tr>
      <th>16</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.00</td>
      <td>-0.138152</td>
      <td>-0.185974</td>
      <td>-0.089611</td>
    </tr>
    <tr>
      <th>17</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.25</td>
      <td>-0.176623</td>
      <td>-0.244273</td>
      <td>-0.107141</td>
    </tr>
  </tbody>
</table>
</div>




```python
# average by both educ4 and dist100
bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    conditional=None,
    average_by=["educ4", "dist100"]
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
      <th>educ4</th>
      <th>dist100</th>
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.00</td>
      <td>0.00591</td>
      <td>-0.085861</td>
      <td>-0.109133</td>
      <td>-0.061614</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.00</td>
      <td>0.02409</td>
      <td>-0.096272</td>
      <td>-0.127518</td>
      <td>-0.069670</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.00</td>
      <td>0.02454</td>
      <td>-0.056617</td>
      <td>-0.065433</td>
      <td>-0.046970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.00</td>
      <td>0.02791</td>
      <td>-0.097646</td>
      <td>-0.128131</td>
      <td>-0.069660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>0.00</td>
      <td>0.03252</td>
      <td>-0.076300</td>
      <td>-0.095832</td>
      <td>-0.057900</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2992</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.00</td>
      <td>1.13727</td>
      <td>-0.070078</td>
      <td>-0.094698</td>
      <td>-0.046623</td>
    </tr>
    <tr>
      <th>2993</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.00</td>
      <td>1.14418</td>
      <td>-0.125547</td>
      <td>-0.172943</td>
      <td>-0.075368</td>
    </tr>
    <tr>
      <th>2994</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.00</td>
      <td>1.25308</td>
      <td>-0.156780</td>
      <td>-0.218836</td>
      <td>-0.088258</td>
    </tr>
    <tr>
      <th>2995</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.00</td>
      <td>1.67025</td>
      <td>-0.161465</td>
      <td>-0.227211</td>
      <td>-0.085394</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>arsenic</td>
      <td>dydx</td>
      <td>4.25</td>
      <td>0.29633</td>
      <td>-0.176623</td>
      <td>-0.244273</td>
      <td>-0.107141</td>
    </tr>
  </tbody>
</table>
<p>2997 rows × 7 columns</p>
</div>



It is still possible to use `plot_slopes` when passing an argument to `average_by`. In the plot below, the empirical distribution is used to compute unit level slopes with respect to `arsenic` and then averaged over `educ4` to obtain the average predictive slopes.


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    conditional=None,
    average_by="educ4"
)
fig.set_size_inches(7, 3)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_39_0.png)
    


### Interpreting coefficients as an elasticity

In some fields, such as economics, it is useful to interpret the results of a regression model in terms of an elasticity (a percent change in $x$ is associated with a percent change in $y$) or semi-elasticity (a unit change in $x$ is associated with a percent change in $y$, or vice versa). Typically, this is achieved by fitting a model where either the outcome and or the covariates are log-transformed. However, since the log transformation is performed by the modeler, to compute elasticities for `slopes` and `plot_slopes`, Bambi "post-processes" the predictions to compute the elasticities. Below, it is shown the possible elasticity arguments and how they are computed for `slopes` and `plot_slopes`:

- `eyex`: a percentage point increase in $x_1$ is associated with an $n$ percentage point increase in $y$

$$\frac{\partial \hat{y}}{\partial x_1} * \frac{x_1}{\hat{y}}$$

- `eydx`: a unit increase in $x_1$ is associated with an $n$ percentage point increase in $y$

$$\frac{\partial \hat{y}}{\partial x_1} * \frac{1}{\hat{y}}$$

- `dyex`: a percentage point increase in $x_1$ is associated with an $n$ unit increase in $y$

$$\frac{\partial \hat{y}}{\partial x_1} * x_1$$

Below, each code cell shows the same model, and `wrt` and `conditional` argument, but with a different elasticity (`slope`) argument. By default, `dydx` (a derivative with no post-processing) is used.


```python
bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    slope="eyex",
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
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>eyex</td>
      <td>-0.525124</td>
      <td>-0.652708</td>
      <td>-0.396082</td>
    </tr>
  </tbody>
</table>
</div>




```python
bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    slope="eydx",
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
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>eydx</td>
      <td>-0.286753</td>
      <td>-0.351592</td>
      <td>-0.220459</td>
    </tr>
  </tbody>
</table>
</div>




```python
bmb.interpret.slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    slope="dyex",
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
      <th>estimate</th>
      <th>lower_3.0%</th>
      <th>upper_97.0%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arsenic</td>
      <td>dyex</td>
      <td>-0.167616</td>
      <td>-0.201147</td>
      <td>-0.134605</td>
    </tr>
  </tbody>
</table>
</div>



`slope` is also an argument for `plot_slopes`. Below, we visualize the elasticity with respect to `arsenic` conditional on a range of `dist100` and `educ4` values (notice this is the same plot as in the _conditional slopes_ section).


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model_interact,
    well_idata_interact,
    wrt="arsenic",
    conditional=["dist100", "educ4"],
    slope="eyex",
    subplot_kwargs={"main": "dist100", "group": "educ4", "panel": "educ4"},
    fig_kwargs={"figsize": (16, 4), "sharey": True},
    legend=False
)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_45_0.png)
    


### Categorical covariates

As mentioned in the _computing slopes_ section, if you pass a variable with a string or categorical data type, the `comparisons` function will be called to compute the expected difference in group means. Here, we fit the same interaction model as above, albeit, by specifying `educ4` as an ordinal data type.


```python
data = pd.read_csv("http://www.stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat", sep=" ")
data["switch"] = pd.Categorical(data["switch"])
data["dist100"] = data["dist"] / 100
data["educ4"] = pd.Categorical(data["educ"] / 4, ordered=True)
```


```python
well_model_interact = bmb.Model(
    "switch ~ dist100 + arsenic + educ4 + dist100:educ4 + arsenic:educ4",
    data,
    family="bernoulli"
)

well_idata_interact = well_model_interact.fit(
    draws=1000, 
    target_accept=0.95, 
    random_seed=1234, 
    chains=4
)
```


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model_interact,
    well_idata_interact,
    wrt="educ4",
    conditional="dist100",
    average_by="dist100"
)
fig.set_size_inches(7, 3)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_49_0.png)
    


As the model was fit with `educ4` as a categorical data type, Bambi recognized this, and calls `comparisons` to compute the differences between each level of `educ4`. As `educ4` contains many category levels, a covariate must be passed to `average_by` in order to perform plotting. Below, we can see this plot is equivalent to `plot_comparisons`.


```python
fig, ax = bmb.interpret.plot_comparisons(
    well_model_interact,
    well_idata_interact,
    contrast="educ4",
    conditional="dist100",
    average_by="dist100"
)
fig.set_size_inches(7, 3)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_51_0.png)
    


However, computing the predictive difference between each `educ4` level may not be desired. Thus, in `plot_slopes`, as in `plot_comparisons`, if `wrt` is a categorical or string data type, it is possible to specify the `wrt` values. For example, if we wanted to compute the expected difference in probability of switching wells for when `educ4` is $4$ versus $1$ conditional on a range of `dist100` and `arsenic` values, we would pass the following dictionary in the code block below. Please refer to the [comparisons](https://bambinos.github.io/bambi/notebooks/plot_comparisons.html) documentation for more details.


```python
fig, ax = bmb.interpret.plot_slopes(
    well_model_interact,
    well_idata_interact,
    wrt={"educ4": [1, 4]},
    conditional="dist100",
    average_by="dist100"
)
fig.set_size_inches(7, 3)
```


    
![png](2023-gsoc-update-slopes_files/2023-gsoc-update-slopes_53_0.png)
    



```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Wed Aug 16 2023
    
    Python implementation: CPython
    Python version       : 3.11.0
    IPython version      : 8.13.2
    
    pandas: 2.0.1
    arviz : 0.15.1
    bambi : 0.10.0.dev0
    
    Watermark: 2.3.1
    

