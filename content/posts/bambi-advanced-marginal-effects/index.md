---
title: 'Advanced Interpret Usage in Bambi'
date: 2023-12-09
author: 'Gabriel Stechschulte'
draft: false
categories: ['probabilistic-programming']
---

<!--eofm-->

# Interpret Advanced Usage

The `interpret` module is inspired by the R package [marginaleffects](https://marginaleffects.com) and ports the core functionality of {marginaleffects} to Bambi. To close the gap of non-supported functionality in Bambi, `interpret` now provides a set of helper functions to aid the user in more advanced and complex analysis not covered within the `comparisons`, `predictions`, and `slopes` functions. 

These helper functions are `data_grid` and `select_draws`. The `data_grid` can be used to create a pairwise grid of data points for the user to pass to `model.predict`. Subsequently, `select_draws` is used to select the draws from the posterior (or posterior predictive) group of the InferenceData object returned by the predict method that correspond to the data points that "produced" that draw.

With access to the appropriately indexed draws, and data used to generate those draws, it enables for more complex analysis such as cross-comparisons and the choice of which model parameter to compute a quantity of interest for. Additionally, the user has more control over the data passed to `model.predict`. Below, it will be demonstrated how to use these helper functions. First, to reproduce the results from the standard `interpret` API, and then to compute cross-comparisons.


```python
import warnings

import arviz as az
import numpy as np
import pandas as pd

import bambi as bmb

from bambi.interpret.helpers import data_grid, select_draws

warnings.simplefilter(action='ignore', category=FutureWarning)
```

## Zero Inflated Poisson

We will adopt the zero inflated Poisson (ZIP) model from the [comparisons](https://bambinos.github.io/bambi/notebooks/plot_comparisons.html)  documentation to demonstrate the helper functions introduced above. 

The ZIP model will be used to predict how many fish are caught by visitors at a state park using survey [data](http://www.stata-press.com/data/r11/fish.dta). Many visitors catch zero fish, either because they did not fish at all, or because they were unlucky. We would like to explicitly model this bimodal behavior (zero versus non-zero) using a Zero Inflated Poisson model, and to compare how different inputs of interest $w$ and other covariate values $c$ are associated with the number of fish caught. The dataset contains data on 250 groups that went to a state park to fish. Each group was questioned about how many fish they caught (`count`), how many children were in the group (`child`), how many people were in the group (`persons`), if they used a live bait and whether or not they brought a camper to the park (`camper`).


```python
fish_data = pd.read_stata("http://www.stata-press.com/data/r11/fish.dta")
cols = ["count", "livebait", "camper", "persons", "child"]
fish_data = fish_data[cols]
fish_data["child"] = fish_data["child"].astype(np.int8)
fish_data["persons"] = fish_data["persons"].astype(np.int8)
fish_data["livebait"] = pd.Categorical(fish_data["livebait"])
fish_data["camper"] = pd.Categorical(fish_data["camper"])
```


```python
fish_model = bmb.Model(
    "count ~ livebait + camper + persons + child", 
    fish_data, 
    family='zero_inflated_poisson'
)

fish_idata = fish_model.fit(random_seed=1234)
```

## Create a grid of data

`data_grid` allows you to create a pairwise grid, also known as a cross-join or cartesian product, of data using the covariates passed to the `conditional` and the optional `variable` parameter. Covariates not passed to `conditional`, but are terms in the Bambi model, are set to typical values (e.g., mean or mode). If you are coming from R, this function is partially inspired from the [data_grid](https://modelr.tidyverse.org/reference/data_grid.html) function in {modelr}. 

There are two ways to create a pairwise grid of data:

1. user-provided values are passed as a dictionary to `conditional` where the keys are the names of the covariates and the values are the values to use in the grid.
2. a list of covariates where the elements are the names of the covariates to use in the grid. As only the names of the covariates were passed, default values are computed to construct the grid.

Any unspecified covariates, i.e., covariates not passed to `conditional` but are terms in the Bambi model, are set to their "typical" values such as mean or mode depending on the data type of the covariate.

### User-provided values

To construct a pairwise grid of data for specific covariate values, pass a dictionary to `conditional`. The values of the dictionary can be of type `int`, `float`, `list`, or `np.ndarray`.


```python
conditional = {
    "camper": np.array([0, 1]),
    "persons": np.arange(1, 5, 1),
    "child": np.array([1, 2, 3]),
}
user_passed_grid = data_grid(fish_model, conditional)
user_passed_grid.query("camper == 0")
```

    Default computed for unspecified variable: livebait





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
      <th>camper</th>
      <th>persons</th>
      <th>child</th>
      <th>livebait</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Subsetting by `camper = 0`, it can be seen that a combination of all possible pairs of values from the dictionary (including the unspecified variable `livebait`) results in a dataframe containing every possible combination of values from the original sets. `livebait` has been set to 1 as this is the mode of the unspecified categorical variable.

### Default values

Alternatively, a list of covariates can be passed to `conditional` where the elements are the names of the covariates to use in the grid. By doing this, you are telling `interpret` to compute default values for these covariates. The psuedocode below outlines the logic and functions used to compute these default values:

```python
if is_numeric_dtype(x) or is_float_dtype(x):
    values = np.linspace(np.min(x), np.max(x), 50)

elif is_integer_dtype(x):
    values = np.quantile(x, np.linspace(0, 1, 5))

elif is_categorical_dtype(x) or is_string_dtype(x) or is_object_dtype(x):
    values = np.unique(x)
```


```python
conditional = ["camper", "persons", "child"]
default_grid = data_grid(fish_model, conditional)

default_grid.shape, user_passed_grid.shape
```

    Default computed for conditional variable: camper, persons, child
    Default computed for unspecified variable: livebait





    ((32, 4), (24, 4))



Notice how the resulting length is different between the user passed and default grid. This is due to the fact that values for `child` range from 0 to 3 for the default grid.


```python
default_grid.query("camper == 0")
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
      <th>camper</th>
      <th>persons</th>
      <th>child</th>
      <th>livebait</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>2</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>3</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>4</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>4</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Compute comparisons

To use `data_grid` to help generate data in computing comparisons or slopes, additional data is passed to the optional `variable` parameter. The name `variable` is an abstraction for the comparisons parameter `contrast` and slopes parameter `wrt`. If you have used any of the `interpret` functions, these parameter names should be familiar and the use of `data_grid` should be analogous to `comparisons`, `predictions`, and `slopes`.

`variable` can also be passed user-provided data (as a dictionary), or a string indicating the name of the covariate of interest. If the latter, a default value will be computed. Additionally, if an argument is passed for `variable`, then the `effect_type` needs to be passed. This is because for `comparisons` and `slopes` an epsilon value `eps` needs to be determined to compute the centered and finite difference, respectively. You can also pass a value for `eps` as a kwarg.


```python
conditional = {
    "camper": np.array([0, 1]),
    "persons": np.arange(1, 5, 1),
    "child": np.array([1, 2, 3, 4])
}
variable = "livebait"

grid = data_grid(fish_model, conditional, variable, effect_type="comparisons")
```

    Default computed for contrast variable: livebait



```python
idata_grid = fish_model.predict(fish_idata, data=grid, inplace=False)
```

### Select draws conditional on data

The second helper function to aid in more advanced analysis is `select_draws`. This is a function that selects the posterior or posterior predictive draws from the ArviZ [InferenceData](https://python.arviz.org/en/stable/getting_started/WorkingWithInferenceData.html) object returned by `model.predict` given a conditional dictionary. The conditional dictionary represents the values that correspond to that draw. 

For example, if we wanted to select posterior draws where `livebait = [0, 1]`, then all we need to do is pass a dictionary where the key is the name of the covariate and the value is the value that we want to condition on (or select). The resulting InferenceData object will contain the draws that correspond to the data points where `livebait = [0, 1]`. Additionally, you must pass the InferenceData object returned by `model.predict`, the data used to generate the predictions, and the name of the data variable `data_var` you would like to select from the InferenceData posterior group. If you specified to return the posterior predictive samples by passing `model.predict(..., kind="pps")`, you can use this group instead of the posterior group by passing `group="posterior_predictive"`.

Below, it is demonstrated how to compute comparisons for `count_mean` for the contrast `livebait = [0, 1]` using the posterior draws.


```python
draw_1 = select_draws(idata_grid, grid, {"livebait": 0}, "count_mean")
```


```python
draw_1 = select_draws(idata_grid, grid, {"livebait": 0}, "count_mean")
draw_2 = select_draws(idata_grid, grid, {"livebait": 1}, "count_mean")

comparison_mean = (draw_2 - draw_1).mean(("chain", "draw"))
comparison_hdi = az.hdi(draw_2 - draw_1)

comparison_df = pd.DataFrame(
    {
        "mean": comparison_mean.values,
        "hdi_low": comparison_hdi.sel(hdi="lower")["count_mean"].values,
        "hdi_high": comparison_hdi.sel(hdi="higher")["count_mean"].values,
    }
)
comparison_df.head(10)
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
      <th>hdi_low</th>
      <th>hdi_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.214363</td>
      <td>0.144309</td>
      <td>0.287735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.053678</td>
      <td>0.029533</td>
      <td>0.077615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.013558</td>
      <td>0.006332</td>
      <td>0.021971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.003454</td>
      <td>0.001132</td>
      <td>0.006040</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.512709</td>
      <td>0.369741</td>
      <td>0.661034</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.128316</td>
      <td>0.077068</td>
      <td>0.181741</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.032392</td>
      <td>0.015553</td>
      <td>0.050690</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.008247</td>
      <td>0.003047</td>
      <td>0.014382</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.228708</td>
      <td>0.913514</td>
      <td>1.533121</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.307342</td>
      <td>0.192380</td>
      <td>0.426808</td>
    </tr>
  </tbody>
</table>
</div>



We can compare this comparison with `bmb.interpret.comparisons`.


```python
summary_df = bmb.interpret.comparisons(
    fish_model,
    fish_idata,
    contrast={"livebait": [0, 1]},
    conditional=conditional
)
summary_df.head(10)
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
      <th>camper</th>
      <th>persons</th>
      <th>child</th>
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
      <td>(0, 1)</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.214363</td>
      <td>0.144309</td>
      <td>0.287735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0.053678</td>
      <td>0.029533</td>
      <td>0.077615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0.013558</td>
      <td>0.006332</td>
      <td>0.021971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0.003454</td>
      <td>0.001132</td>
      <td>0.006040</td>
    </tr>
    <tr>
      <th>4</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.512709</td>
      <td>0.369741</td>
      <td>0.661034</td>
    </tr>
    <tr>
      <th>5</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.128316</td>
      <td>0.077068</td>
      <td>0.181741</td>
    </tr>
    <tr>
      <th>6</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0.032392</td>
      <td>0.015553</td>
      <td>0.050690</td>
    </tr>
    <tr>
      <th>7</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0.008247</td>
      <td>0.003047</td>
      <td>0.014382</td>
    </tr>
    <tr>
      <th>8</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1.228708</td>
      <td>0.913514</td>
      <td>1.533121</td>
    </tr>
    <tr>
      <th>9</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0.307342</td>
      <td>0.192380</td>
      <td>0.426808</td>
    </tr>
  </tbody>
</table>
</div>



Albeit the other information in the `summary_df`, the columns `estimate`, `lower_3.0%`, `upper_97.0%` are identical.

### Cross comparisons

Computing a cross-comparison is useful for when we want to compare contrasts when two (or more) predictors change at the same time. Cross-comparisons are currently not supported in the `comparisons` function, but we can use `select_draws` to compute them. For example, imagine we are interested in computing the cross-comparison between the two rows below.


```python
summary_df.iloc[:2]
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
      <th>camper</th>
      <th>persons</th>
      <th>child</th>
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
      <td>(0, 1)</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.214363</td>
      <td>0.144309</td>
      <td>0.287735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>livebait</td>
      <td>diff</td>
      <td>(0, 1)</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0.053678</td>
      <td>0.029533</td>
      <td>0.077615</td>
    </tr>
  </tbody>
</table>
</div>



The cross-comparison amounts to first computing the comparison for row 0, given below, and can be verified by looking at the estimate in `summary_df`.


```python
cond_10 = {
    "camper": 0,
    "persons": 1,
    "child": 1,
    "livebait": 0 
}

cond_11 = {
    "camper": 0,
    "persons": 1,
    "child": 1,
    "livebait": 1
}

draws_10 = select_draws(idata_grid, grid, cond_10, "count_mean")
draws_11 = select_draws(idata_grid, grid, cond_11, "count_mean")

(draws_11 - draws_10).mean(("chain", "draw")).item()
```




    0.2143627093182434



Next, we need to compute the comparison for row 1.


```python
cond_20 = {
    "camper": 0,
    "persons": 1,
    "child": 2,
    "livebait": 0
}

cond_21 = {
    "camper": 0,
    "persons": 1,
    "child": 2,
    "livebait": 1
}

draws_20 = select_draws(idata_grid, grid, cond_20, "count_mean")
draws_21 = select_draws(idata_grid, grid, cond_21, "count_mean")
```


```python
(draws_21 - draws_20).mean(("chain", "draw")).item()
```




    0.053678256991883604



Next, we compute the "first level" comparisons (`diff_1` and `diff_2`). Subsequently, we compute the difference between these two differences to obtain the cross-comparison.


```python
diff_1 = (draws_11 - draws_10)
diff_2 = (draws_21 - draws_20)

cross_comparison = (diff_2 - diff_1).mean(("chain", "draw")).item()
cross_comparison
```




    -0.16068445232635978



To verify this is correct, we can check by performing the cross-comparison directly on the `summary_df`.


```python
summary_df.iloc[1]["estimate"] - summary_df.iloc[0]["estimate"]
```




    -0.16068445232635978



## Summary

In this notebook, the interpret helper functions `data_grid` and `select_draws` were introduced and it was demonstrated how they can be used to compute pairwise grids of data and cross-comparisons. With these functions, it is left to the user to generate their grids of data and quantities of interest allowing for more flexibility and control over the type of data passed to `model.predict` and the quantities of interest computed.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Tue Dec 05 2023
    
    Python implementation: CPython
    Python version       : 3.11.0
    IPython version      : 8.13.2
    
    numpy : 1.24.2
    pandas: 2.1.0
    bambi : 0.13.0.dev0
    arviz : 0.16.1
    
    Watermark: 2.3.1
    

