---
title: 'Predict New Groups with Hierarchical Models in Bambi'
date: 2023-10-10
author: 'Gabriel Stechschulte'
draft: false
math: true
categories: ['probabilistic-programming']
---

<!--eofm-->

# Predict New Groups

In Bambi, it is possible to perform predictions on new, unseen, groups of data that were not in the observed data used to fit the model with the argument `sample_new_groups` in the `model.predict()` method. This is useful in the context of hierarchical modeling, where groups are assumed to be a sample from a larger group.

This blog post is a copy of the zero inflated models documentation I wrote for [Bambi](https://bambinos.github.io/bambi/). The original post can be found [here](https://bambinos.github.io/bambi/notebooks/predict_new_groups.html).

Below, it is first described how predictions at multiple levels and for unseen groups are possible with hierarchical models. Then, it is described how this is performed in Bambi. Lastly, a hierarchical model is developed to show how to use the `sample_new_groups` argument in the `model.predict()` method, and within the `interpret` sub-package. For users coming from `brms` in R, this is equivalent to the `sample_new_levels` argument.

## Hierarchical models and predictions at multiple levels

A feature of hierarchical models is that they are able to make predictions at multiple levels. For example, if we were to use the penguin [dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris) to fit a hierchical regression to estimate the body mass of each penguin species given a set of predictors, we could estimate the mass of all penguins **and** each individual species at the same time. Thus, in this example, there are predictions for two levels: (1) the population level, and (2) the species level.

Additionally, a hierarchical model can be used to make predictions for groups (levels) that were never seen before if a hyperprior is defined over the group-specific effect. With a hyperior defined on group-specific effects, the groups do not share one fixed parameter, but rather share a hyperprior distribution which describes the distribution for the parameter of the prior itself. Lets write a hierarchical model (without intercepts) with a hyperprior defined for group-specific effects in statistical notation so this concept becomes more clear:

$$\beta_{\mu h} \sim \mathcal{N}(0, 10)$$
$$\beta_{\sigma h} \sim \mathcal{HN}(10)$$
$$\beta_{m} \sim \mathcal{N}(\beta_{\mu h}, \beta_{\sigma h})$$
$$\sigma_{h} \sim \mathcal{HN}(10)$$
$$\sigma_{m} \sim \mathcal{HN}(\sigma_{h})$$
$$Y \sim \mathcal{N}(\beta_{m} * X_{m}, \sigma_{m})$$

The parameters $\beta_{\mu h}, \beta_{\sigma h}$ of the group-specific effect prior $\beta_{m}$ come from hyperprior distributions. Thus, if we would like to make predictions for a new, unseen, group, we can do so by first sampling from these hyperprior distributions to obtain the parameters for the new group, and then sample from the posterior or posterior predictive distribution to obtain the estimates for the new group. For a more in depth explanation of hierarchical models in Bambi, see either: the [radon example](https://bambinos.github.io/bambi/notebooks/radon_example.html#varying-intercept-and-slope-model), or the [sleep study example](https://bambinos.github.io/bambi/notebooks/sleepstudy.html).

## Sampling new groups in Bambi

If data with unseen groups are passed to the `new_data` argument of the `model.predict()` method, Bambi first needs to identify if that group exists, and if not, to evaluate the new group with the respective group-specific term. This evaluation updates the [design matrix](https://bambinos.github.io/bambi/notebooks/how_bambi_works.html#design-matrix) initially used to fit the model with the new group(s). This is achieved with the `.evaluate_new_data` method in the [formulae](https://github.com/bambinos/formulae) package.

Once the design matrix has been updated, Bambi can perform predictions on the new, unseen, groups by specifying `sample_new_groups=True` in `model.predict()`. Each posterior sample for the new groups is drawn from the posterior draws of a randomly selected existing group. Since different groups may be selected at each draw, the end result represents the variation across existing groups.

## Hierarchical regression

To demonstrate the `sample_new_groups` argument, we will develop a hierarchical model on the [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression) dataset. Pulmonary fibrosis is a disorder with no known cause and no known cure, created by scarring of the lungs. Using a hierarchical model, the objective is to predict a patient’s severity of decline in lung function. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (FVC), i.e. the volume of air exhaled by the patient.


```python
#| code-fold: true
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

import bambi as bmb

warnings.simplefilter(action="ignore", category=FutureWarning)
```

### The OSIC pulmonary fibrosis progression dataset

In the dataset, we were provided with a baseline chest computerized tomography (CT) scan and associated clinical information for a set of patients where the columns represent the following

 - `patient`- a unique id for each patient
 - `weeks`- the relative number of weeks pre/post the baseline CT (may be negative)
 - `fvc` - the recorded lung capacity in millilitres (ml)
 - `percent`- a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
 - `sex` - male or female
 - `smoking_status` - ex-smoker, never smoked, currently smokes
 - `age` - age of the patient

A patient has an image acquired at time `week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their FVC is measured. Below, we randomly sample three patients and plot their FVC measurements over time.


```python
data = pd.read_csv(
    "https://gist.githubusercontent.com/ucals/"
    "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
    "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
    "osic_pulmonary_fibrosis.csv"
)

data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace("smokingstatus", "smoking_status")
data
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
      <th>patient</th>
      <th>weeks</th>
      <th>fvc</th>
      <th>percent</th>
      <th>age</th>
      <th>sex</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID00007637202177411956430</td>
      <td>-4</td>
      <td>2315</td>
      <td>58.253649</td>
      <td>79</td>
      <td>Male</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID00007637202177411956430</td>
      <td>5</td>
      <td>2214</td>
      <td>55.712129</td>
      <td>79</td>
      <td>Male</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID00007637202177411956430</td>
      <td>7</td>
      <td>2061</td>
      <td>51.862104</td>
      <td>79</td>
      <td>Male</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ID00007637202177411956430</td>
      <td>9</td>
      <td>2144</td>
      <td>53.950679</td>
      <td>79</td>
      <td>Male</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ID00007637202177411956430</td>
      <td>11</td>
      <td>2069</td>
      <td>52.063412</td>
      <td>79</td>
      <td>Male</td>
      <td>Ex-smoker</td>
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
      <th>1544</th>
      <td>ID00426637202313170790466</td>
      <td>13</td>
      <td>2712</td>
      <td>66.594637</td>
      <td>73</td>
      <td>Male</td>
      <td>Never smoked</td>
    </tr>
    <tr>
      <th>1545</th>
      <td>ID00426637202313170790466</td>
      <td>19</td>
      <td>2978</td>
      <td>73.126412</td>
      <td>73</td>
      <td>Male</td>
      <td>Never smoked</td>
    </tr>
    <tr>
      <th>1546</th>
      <td>ID00426637202313170790466</td>
      <td>31</td>
      <td>2908</td>
      <td>71.407524</td>
      <td>73</td>
      <td>Male</td>
      <td>Never smoked</td>
    </tr>
    <tr>
      <th>1547</th>
      <td>ID00426637202313170790466</td>
      <td>43</td>
      <td>2975</td>
      <td>73.052745</td>
      <td>73</td>
      <td>Male</td>
      <td>Never smoked</td>
    </tr>
    <tr>
      <th>1548</th>
      <td>ID00426637202313170790466</td>
      <td>59</td>
      <td>2774</td>
      <td>68.117081</td>
      <td>73</td>
      <td>Male</td>
      <td>Never smoked</td>
    </tr>
  </tbody>
</table>
<p>1549 rows × 7 columns</p>
</div>




```python
def label_encoder(labels):
    """
    Encode patient IDs as integers.
    """
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = labels.map(label_to_index)
    return encoded_labels
```


```python
predictors = ["patient", "weeks", "fvc", "smoking_status"]

data["patient"] = label_encoder(data['patient'])

data["weeks"] = (data["weeks"] - data["weeks"].min()) / (
    data["weeks"].max() - data["weeks"].min()
)
data["fvc"] = (data["fvc"] - data["fvc"].min()) / (
    data["fvc"].max() - data["fvc"].min()
)

data = data[predictors]
```


```python
patient_id = data.sample(n=3, random_state=42)["patient"].values

fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i, p in enumerate(patient_id):
    patient_data = data[data["patient"] == p]
    ax[i].scatter(patient_data["weeks"], patient_data["fvc"])
    ax[i].set_xlabel("weeks")
    ax[i].set_ylabel("fvc")
    ax[i].set_title(f"patient {p}")

plt.tight_layout()
```



![png](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_10_0.png)



The plots show variability in FVC measurements, unequal time intervals between follow up visits, and different number of visits per patient. This is a good scenario to use a hierarchical model, where we can model the FVC measurements for each patient as a function of time, and also model the variability in the FVC measurements across patients.

### Partial pooling model

The hierarchical model we will develop is a partially pooled model using the predictors `weeks`, `smoking_status`, and `patient` to predict the response `fvc`. We will estimate the following model with common and group-effects:

- common-effects: `weeks` and `smoking_status`
- group-effects: the slope of `weeks` will vary by `patient`

Additionally, the global intercept is not included. Since the global intercept is excluded, `smoking_status` uses cell means encoding (i.e. the coefficient represents the estimate for each `smoking_status` category of the entire group). This logic also applies for `weeks`. However, a group-effect is also specified for `weeks`, which means that the association between `weeks` and the `fvc` is allowed to vary by individual `patients`.

Below, the default prior for the group-effect sigma is changed from `HalfNormal` to a `Gamma` distribution. Additionally, the model graph shows the model has been reparameterized to be non-centered. This is the default when there are group-effects in Bambi.


```python
priors = {
    "weeks|patient": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Gamma", alpha=3, beta=3)),
}

model = bmb.Model(
    "fvc ~ 0 + weeks + smoking_status + (0 + weeks | patient)",
    data,
    priors=priors,
    categorical=["patient", "smoking_status"],
)
model.build()
model.graph()
```





![svg](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_13_0.svg)





```python
idata = model.fit(
    draws=1500,
    tune=1000,
    target_accept=0.95,
    chains=4,
    random_seed=42,
    cores=10,
)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 10 jobs)
    NUTS: [fvc_sigma, weeks, smoking_status, weeks|patient_sigma, weeks|patient_offset]




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
  <progress value='10000' class='' max='10000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [10000/10000 00:10&lt;00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_500 draw iterations (4_000 + 6_000 draws total) took 10 seconds.


### Model criticism

Hierarchical models can induce difficult posterior geometries to sample from. Below, we quickly analyze the traces to ensure sampling went well.


```python
az.plot_trace(idata)
plt.tight_layout();
```



![png](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_16_0.png)



Analyzing the marginal posteriors of `weeks` and `weeks|patient`, we see that the slope can be very different for some individuals. `weeks` indicates that as a population, the slope is negative. However, `weeks|patients` indicates some patients are negative, some are positive, and some are close to zero. Moreover, there are varying levels of uncertainty observed in the coefficients for the three different values of the `smoking_status` variable.


```python
az.summary(idata, var_names=["weeks", "smoking_status", "fvc_sigma", "weeks|patient_sigma"])
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
      <th>weeks</th>
      <td>-0.116</td>
      <td>0.036</td>
      <td>-0.183</td>
      <td>-0.048</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>447.0</td>
      <td>836.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>smoking_status[Currently smokes]</th>
      <td>0.398</td>
      <td>0.017</td>
      <td>0.364</td>
      <td>0.429</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3315.0</td>
      <td>4274.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>smoking_status[Ex-smoker]</th>
      <td>0.382</td>
      <td>0.005</td>
      <td>0.373</td>
      <td>0.392</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5104.0</td>
      <td>4835.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>smoking_status[Never smoked]</th>
      <td>0.291</td>
      <td>0.007</td>
      <td>0.277</td>
      <td>0.305</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3299.0</td>
      <td>4290.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>fvc_sigma</th>
      <td>0.077</td>
      <td>0.001</td>
      <td>0.074</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>8405.0</td>
      <td>4362.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>weeks|patient_sigma</th>
      <td>0.458</td>
      <td>0.027</td>
      <td>0.412</td>
      <td>0.511</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>767.0</td>
      <td>1559.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</div>



The effective sample size (ESS) is much lower for the `weeks` and `weeks|patient_sigma` parameters. This can also be inferred visually by looking at the trace plots for these parameters above. There seems to be some autocorrelation in the samples for these parameters. However, for the sake of this example, we will not worry about this.

### Predict observed patients

First, we will use the posterior distribution to plot the mean and 95% credible interval for the FVC measurements of the three randomly sampled patients above.


```python
preds = model.predict(idata, kind="mean", inplace=False)
fvc_mean = az.extract(preds["posterior"])["fvc_mean"]
```


```python
# plot posterior predictions
fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i, p in enumerate(patient_id):
    idx = data.index[data["patient"] == p].tolist()
    weeks = data.loc[idx, "weeks"].values
    fvc = data.loc[idx, "fvc"].values

    ax[i].scatter(weeks, fvc)
    az.plot_hdi(weeks, fvc_mean[idx].T, color="C0", ax=ax[i])
    ax[i].plot(weeks, fvc_mean[idx].mean(axis=1), color="C0")

    ax[i].set_xlabel("weeks")
    ax[i].set_ylabel("fvc")
    ax[i].set_title(f"patient {p}")

plt.tight_layout()
```



![png](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_22_0.png)



The plots show that the posterior estimates seem to fit the three patients well. Where there are more observations, the credible interval is smaller, and where there are fewer observations, the credible interval is larger. Next, we will predict new, unseen, patients.

### Predict new patients

Imagine the cost of acquiring a CT scan increases dramatically, and we would like to interopolate the FVC measurement for a new patient with a given set of clinical information `smoking_status` and `weeks`. We achieve this by passing this data to the predict method and setting `sample_new_groups=True`. As outlined in the _Sampling new groups in Bambi_ section, this new data is evaluated by `formulae` to update the design matrix, and then predictions are made for the new group by sampling from the posterior draws of a randomly selected existing group.

Below, we will simulate a new patient and predict their FVC measurements over time. First, we will copy clinical data from patient 39 and use it for patient 176 (the new, unseen, patient). Subsequently, we will construct another new patient, with different clinical data.


```python
# copy patient 39 data to the new patient 176
patient_39 = data[data["patient"] == 39].reset_index(drop=True)
new_data = patient_39.copy()
new_data["patient"] = 176
new_data = pd.concat([new_data, patient_39]).reset_index(drop=True)[predictors]
new_data
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
      <th>patient</th>
      <th>weeks</th>
      <th>fvc</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>176</td>
      <td>0.355072</td>
      <td>0.378141</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>1</th>
      <td>176</td>
      <td>0.376812</td>
      <td>0.365937</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>2</th>
      <td>176</td>
      <td>0.391304</td>
      <td>0.401651</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>3</th>
      <td>176</td>
      <td>0.405797</td>
      <td>0.405958</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>4</th>
      <td>176</td>
      <td>0.420290</td>
      <td>0.390883</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>5</th>
      <td>176</td>
      <td>0.456522</td>
      <td>0.390165</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>6</th>
      <td>176</td>
      <td>0.543478</td>
      <td>0.348528</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>7</th>
      <td>176</td>
      <td>0.637681</td>
      <td>0.337581</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>8</th>
      <td>176</td>
      <td>0.746377</td>
      <td>0.365219</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>9</th>
      <td>176</td>
      <td>0.775362</td>
      <td>0.360014</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>10</th>
      <td>39</td>
      <td>0.355072</td>
      <td>0.378141</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>11</th>
      <td>39</td>
      <td>0.376812</td>
      <td>0.365937</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>12</th>
      <td>39</td>
      <td>0.391304</td>
      <td>0.401651</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>13</th>
      <td>39</td>
      <td>0.405797</td>
      <td>0.405958</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>14</th>
      <td>39</td>
      <td>0.420290</td>
      <td>0.390883</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>15</th>
      <td>39</td>
      <td>0.456522</td>
      <td>0.390165</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>16</th>
      <td>39</td>
      <td>0.543478</td>
      <td>0.348528</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>17</th>
      <td>39</td>
      <td>0.637681</td>
      <td>0.337581</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>18</th>
      <td>39</td>
      <td>0.746377</td>
      <td>0.365219</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>19</th>
      <td>39</td>
      <td>0.775362</td>
      <td>0.360014</td>
      <td>Ex-smoker</td>
    </tr>
  </tbody>
</table>
</div>




```python
preds = model.predict(
    idata, kind="mean",
    data=new_data,
    sample_new_groups=True,
    inplace=False
)
```


```python
# utility func for plotting
def plot_new_patient(idata, data, patient_ids):
    fvc_mean = az.extract(idata["posterior"])["fvc_mean"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for i, p in enumerate(patient_ids):
        idx = data.index[data["patient"] == p].tolist()
        weeks = data.loc[idx, "weeks"].values
        fvc = data.loc[idx, "fvc"].values

        if p == patient_ids[0]:
            ax[i].scatter(weeks, fvc)

        az.plot_hdi(weeks, fvc_mean[idx].T, color="C0", ax=ax[i])
        ax[i].plot(weeks, fvc_mean[idx].mean(axis=1), color="C0")

        ax[i].set_xlabel("weeks")
        ax[i].set_ylabel("fvc")
        ax[i].set_title(f"patient {p}")
```


```python
plot_new_patient(preds, new_data, [39, 176])
```



![png](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_28_0.png)



Although identical data was used for both patients, the variability increased consideribly for patient 176. However, the mean predictions for both patients appear to be almost identical. Now, lets construct a new patient with different clinical data and see how the predictions change. We will select 10 time of follow up visits at random, and set the `smoking_status = "Currently smokes"`.


```python
new_data.loc[new_data["patient"] == 176, "smoking_status"] = "Currently smokes"
weeks = np.random.choice(sorted(model.data.weeks.unique()), size=10)
new_data.loc[new_data["patient"] == 176, "weeks"] = weeks
new_data
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
      <th>patient</th>
      <th>weeks</th>
      <th>fvc</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>176</td>
      <td>0.173913</td>
      <td>0.378141</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>176</td>
      <td>0.137681</td>
      <td>0.365937</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>176</td>
      <td>0.608696</td>
      <td>0.401651</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>176</td>
      <td>0.717391</td>
      <td>0.405958</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>176</td>
      <td>0.521739</td>
      <td>0.390883</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>176</td>
      <td>0.246377</td>
      <td>0.390165</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>176</td>
      <td>0.181159</td>
      <td>0.348528</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>176</td>
      <td>0.702899</td>
      <td>0.337581</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>176</td>
      <td>0.456522</td>
      <td>0.365219</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>176</td>
      <td>0.333333</td>
      <td>0.360014</td>
      <td>Currently smokes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>39</td>
      <td>0.355072</td>
      <td>0.378141</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>11</th>
      <td>39</td>
      <td>0.376812</td>
      <td>0.365937</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>12</th>
      <td>39</td>
      <td>0.391304</td>
      <td>0.401651</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>13</th>
      <td>39</td>
      <td>0.405797</td>
      <td>0.405958</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>14</th>
      <td>39</td>
      <td>0.420290</td>
      <td>0.390883</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>15</th>
      <td>39</td>
      <td>0.456522</td>
      <td>0.390165</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>16</th>
      <td>39</td>
      <td>0.543478</td>
      <td>0.348528</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>17</th>
      <td>39</td>
      <td>0.637681</td>
      <td>0.337581</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>18</th>
      <td>39</td>
      <td>0.746377</td>
      <td>0.365219</td>
      <td>Ex-smoker</td>
    </tr>
    <tr>
      <th>19</th>
      <td>39</td>
      <td>0.775362</td>
      <td>0.360014</td>
      <td>Ex-smoker</td>
    </tr>
  </tbody>
</table>
</div>



If we were to keep the default value of `sample_new_groups=False`, the following error would be raised: `ValueError: There are new groups for the factors ('patient',) and 'sample_new_groups' is False`. Thus, we set `sample_new_groups=True` and obtain predictions for the new patient.


```python
preds = model.predict(
    idata, kind="mean",
    data=new_data,
    sample_new_groups=True,
    inplace=False
)
```


```python
plot_new_patient(preds, new_data, [39, 176])
```



![png](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_33_0.png)



With `smoking_status = "Currently smokes"`, and the time of follow up visit randomly selected, we can see that the intercept is slightly higher, and it appears that the slope is steeper for this new patient. Again, the variability is much higher for patient 176, and in particular, where there are fewer `fvc` measurements.

#### Predict new patients with `interpret`

The `interpret` sub-package in Bambi allows us to easily interpret the predictions for new patients. In particular, using `bmb.interpret.comparisons`, we can compare the predictions made for a new patient and an existing similar patient. Below, we will compare the predictions made for patient 176 and patient 39. We will use the same clinical data for both patients as we did in the first exampe above.


```python
time_of_follow_up = list(new_data.query("patient == 39")["weeks"].values)
time_of_follow_up
```




    [0.35507246376811596,
     0.37681159420289856,
     0.391304347826087,
     0.4057971014492754,
     0.42028985507246375,
     0.45652173913043476,
     0.5434782608695652,
     0.6376811594202898,
     0.7463768115942029,
     0.7753623188405797]




```python
fig, ax = bmb.interpret.plot_comparisons(
    model,
    idata,
    contrast={"patient": [39, 176]},
    conditional={"weeks": time_of_follow_up, "smoking_status": "Ex-smoker"},
    sample_new_groups=True,
    fig_kwargs={"figsize": (7, 3)}
)
plt.title("Difference in predictions for patient 176 vs 39");
```



![png](2023-10-10-predict-groups-bambi_files/2023-10-10-predict-groups-bambi_37_0.png)



Referring to the plots where patient 39 and 176 use identical data, the mean `fvc` predictions "look" about the same. When this comparison is made quantitatively using the comparisons function, we can see that mean `fvc` measurements are slightly below 0.0, and have a constant slope across `weeks` indicating there is a slight difference in mean `fvc` measurements between the two patients.

## Summary

In this notebook, it was shown how predictions at multiple levels and for unseen groups are possible with hierarchical models. To utilize this feature of hierarchical models, Bambi first updates the design matrix to include the new group. Then, predictions are made for the new group by sampling from the posterior draws of a randomly selected existing group.

To predict new groups in Bambi, you can either: (1) create a dataset with new groups and pass it to the `model.predict()` method while specifying `sample_new_groups=True`, or (2) use the functions `comparisons` or `slopes` in the `interpret` sub-package with `sample_new_groups=True` to compare predictions or slopes for new groups and existing groups.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Tue Oct 10 2023

    Python implementation: CPython
    Python version       : 3.11.0
    IPython version      : 8.13.2

    matplotlib: 3.7.1
    arviz     : 0.16.1
    pandas    : 2.1.0
    numpy     : 1.24.2
    bambi     : 0.13.0.dev0

    Watermark: 2.3.1
