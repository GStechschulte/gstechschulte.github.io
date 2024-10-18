---
title: 'Alternative Samplers to NUTS in Bambi'
date: 2024-03-29
author: 'Gabriel Stechschulte'
draft: false
categories: ['probabilistic-programming']
---

<!--eofm-->

# Alternative sampling backends

This blog post is a copy of the alternative samplers documentation I wrote for [Bambi](https://bambinos.github.io/bambi/). The original post can be found [here](https://bambinos.github.io/bambi/notebooks/).

In Bambi, the sampler used is automatically selected given the type of variables used in the model. For inference, Bambi supports both MCMC and variational inference. By default, Bambi uses PyMC's implementation of the adaptive Hamiltonian Monte Carlo (HMC) algorithm for sampling. Also known as the No-U-Turn Sampler (NUTS). This sampler is a good choice for many models. However, it is not the only sampling method, nor is PyMC the only library implementing NUTS. 

To this extent, Bambi supports multiple backends for MCMC sampling such as NumPyro and Blackjax. This notebook will cover how to use such alternatives in Bambi.

_Note_: Bambi utilizes [bayeux](https://github.com/jax-ml/bayeux) to access a variety of sampling backends. Thus, you will need to install the optional dependencies in the Bambi [pyproject.toml](https://github.com/bambinos/bambi/blob/main/pyproject.toml) file to use these backends.


```python
import arviz as az
import bambi as bmb
import bayeux as bx
import numpy as np
import pandas as pd
```

    WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.


## bayeux

Bambi leverages `bayeux` to access different sampling backends. In short, `bayeux` lets you write a probabilistic model in JAX and immediately have access to state-of-the-art inference methods. 

Since the underlying Bambi model is a PyMC model, this PyMC model can be "given" to `bayeux`. Then, we can choose from a variety of MCMC methods to perform inference. 

To demonstrate the available backends, we will fist simulate data and build a model.


```python
num_samples = 100
num_features = 1
noise_std = 1.0
random_seed = 42

np.random.seed(random_seed)

coefficients = np.random.randn(num_features)
X = np.random.randn(num_samples, num_features)
error = np.random.normal(scale=noise_std, size=num_samples)
y = X @ coefficients + error

data = pd.DataFrame({"y": y, "x": X.flatten()})
```


```python
model = bmb.Model("y ~ x", data)
model.build()
```

We can call `bmb.inference_methods.names` that returns a nested dictionary of the backends and list of inference methods.


```python
methods = bmb.inference_methods.names
methods
```




    {'pymc': {'mcmc': ['mcmc'], 'vi': ['vi']},
     'bayeux': {'mcmc': ['tfp_hmc',
       'tfp_nuts',
       'tfp_snaper_hmc',
       'blackjax_hmc',
       'blackjax_chees_hmc',
       'blackjax_meads_hmc',
       'blackjax_nuts',
       'blackjax_hmc_pathfinder',
       'blackjax_nuts_pathfinder',
       'flowmc_rqspline_hmc',
       'flowmc_rqspline_mala',
       'flowmc_realnvp_hmc',
       'flowmc_realnvp_mala',
       'numpyro_hmc',
       'numpyro_nuts']}}



With the PyMC backend, we have access to their implementation of the NUTS sampler and mean-field variational inference.


```python
methods["pymc"]
```




    {'mcmc': ['mcmc'], 'vi': ['vi']}



`bayeux` lets us have access to Tensorflow probability, Blackjax, FlowMC, and NumPyro backends.


```python
methods["bayeux"]
```




    {'mcmc': ['tfp_hmc',
      'tfp_nuts',
      'tfp_snaper_hmc',
      'blackjax_hmc',
      'blackjax_chees_hmc',
      'blackjax_meads_hmc',
      'blackjax_nuts',
      'blackjax_hmc_pathfinder',
      'blackjax_nuts_pathfinder',
      'flowmc_rqspline_hmc',
      'flowmc_rqspline_mala',
      'flowmc_realnvp_hmc',
      'flowmc_realnvp_mala',
      'numpyro_hmc',
      'numpyro_nuts']}



The values of the MCMC and VI keys in the dictionary are the names of the argument you would pass to `inference_method` in `model.fit`. This is shown in the section below.

## Specifying an `inference_method`

By default, Bambi uses the PyMC NUTS implementation. To use a different backend, pass the name of the `bayeux` MCMC method to the `inference_method` parameter of the `fit` method.

### Blackjax


```python
blackjax_nuts_idata = model.fit(inference_method="blackjax_nuts")
```

Different backends have different naming conventions for the parameters specific to that MCMC method. Thus, to specify backend-specific parameters, pass your own `kwargs` to the `fit` method.

The following can be performend to identify the kwargs specific to each method.


```python
bmb.inference_methods.get_kwargs("blackjax_nuts")
```




    {<function blackjax.adaptation.window_adaptation.window_adaptation(algorithm: Union[blackjax.mcmc.hmc.hmc, blackjax.mcmc.nuts.nuts], logdensity_fn: Callable, is_mass_matrix_diagonal: bool = True, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, progress_bar: bool = False, **extra_parameters) -> blackjax.base.AdaptationAlgorithm>: {'logdensity_fn': <function bayeux._src.shared.constrain.<locals>.wrap_log_density.<locals>.wrapped(args)>,
      'is_mass_matrix_diagonal': True,
      'initial_step_size': 1.0,
      'target_acceptance_rate': 0.8,
      'progress_bar': False,
      'algorithm': blackjax.mcmc.nuts.nuts},
     'adapt.run': {'num_steps': 500},
     blackjax.mcmc.nuts.nuts: {'max_num_doublings': 10,
      'divergence_threshold': 1000,
      'integrator': <function blackjax.mcmc.integrators.generate_euclidean_integrator.<locals>.euclidean_integrator(logdensity_fn: Callable, kinetic_energy_fn: blackjax.mcmc.metrics.KineticEnergy) -> Callable[[blackjax.mcmc.integrators.IntegratorState, float], blackjax.mcmc.integrators.IntegratorState]>,
      'logdensity_fn': <function bayeux._src.shared.constrain.<locals>.wrap_log_density.<locals>.wrapped(args)>,
      'step_size': 0.5},
     'extra_parameters': {'chain_method': 'vectorized',
      'num_chains': 8,
      'num_draws': 500,
      'num_adapt_draws': 500,
      'return_pytree': False}}



Now, we can identify the kwargs we would like to change and pass to the `fit` method.


```python
kwargs = {
        "adapt.run": {"num_steps": 500},
        "num_chains": 4,
        "num_draws": 250,
        "num_adapt_draws": 250
}

blackjax_nuts_idata = model.fit(inference_method="blackjax_nuts", **kwargs)
blackjax_nuts_idata
```

### Tensorflow probability


```python
tfp_nuts_idata = model.fit(inference_method="tfp_nuts")
tfp_nuts_idata
```

### NumPyro


```python
numpyro_nuts_idata = model.fit(inference_method="numpyro_nuts")
numpyro_nuts_idata
```

### flowMC


```python
flowmc_idata = model.fit(inference_method="flowmc_realnvp_hmc")
flowmc_idata
```

## Sampler comparisons

With ArviZ, we can compare the inference result summaries of the samplers. _Note:_ We can't use `az.compare` as not each inference data object returns the pointwise log-probabilities. Thus, an error would be raised.


```python
az.summary(blackjax_nuts_idata)
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
      <td>0.023</td>
      <td>0.097</td>
      <td>-0.141</td>
      <td>0.209</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>694.0</td>
      <td>508.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.356</td>
      <td>0.111</td>
      <td>0.162</td>
      <td>0.571</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>970.0</td>
      <td>675.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>y_sigma</th>
      <td>0.950</td>
      <td>0.069</td>
      <td>0.827</td>
      <td>1.072</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1418.0</td>
      <td>842.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(tfp_nuts_idata)
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
      <td>0.023</td>
      <td>0.097</td>
      <td>-0.157</td>
      <td>0.205</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6785.0</td>
      <td>5740.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.360</td>
      <td>0.105</td>
      <td>0.169</td>
      <td>0.563</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6988.0</td>
      <td>5116.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_sigma</th>
      <td>0.946</td>
      <td>0.067</td>
      <td>0.831</td>
      <td>1.081</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7476.0</td>
      <td>5971.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(numpyro_nuts_idata)
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
      <td>0.024</td>
      <td>0.095</td>
      <td>-0.162</td>
      <td>0.195</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6851.0</td>
      <td>5614.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.362</td>
      <td>0.104</td>
      <td>0.176</td>
      <td>0.557</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>9241.0</td>
      <td>6340.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_sigma</th>
      <td>0.946</td>
      <td>0.068</td>
      <td>0.826</td>
      <td>1.079</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7247.0</td>
      <td>5711.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(flowmc_idata)
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
      <td>0.015</td>
      <td>0.100</td>
      <td>-0.186</td>
      <td>0.190</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>758.0</td>
      <td>1233.0</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.361</td>
      <td>0.105</td>
      <td>0.174</td>
      <td>0.565</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5084.0</td>
      <td>4525.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>y_sigma</th>
      <td>0.951</td>
      <td>0.070</td>
      <td>0.823</td>
      <td>1.079</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5536.0</td>
      <td>5080.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

Thanks to `bayeux`, we can use three different sampling backends and 10+ alternative MCMC methods in Bambi. Using these methods is as simple as passing the inference name to the `inference_method` of the `fit` method.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Sat Apr 13 2024
    
    Python implementation: CPython
    Python version       : 3.12.2
    IPython version      : 8.20.0
    
    bambi : 0.13.1.dev25+g1e7f677e.d20240413
    pandas: 2.2.1
    numpy : 1.26.4
    bayeux: 0.1.10
    arviz : 0.18.0
    
    Watermark: 2.4.3
    

