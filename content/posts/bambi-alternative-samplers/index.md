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
blackjax_nuts_idata
```


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
</pre>







            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">

            <li class = "xr-section-item">
                  <input id="idata_posteriora307c28f-96ce-4b89-96bb-1fe5815237d0" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posteriora307c28f-96ce-4b89-96bb-1fe5815237d0" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 100kB
Dimensions:    (chain: 8, draw: 500)
Coordinates:
  * chain      (chain) int64 64B 0 1 2 3 4 5 6 7
  * draw       (draw) int64 4kB 0 1 2 3 4 5 6 7 ... 493 494 495 496 497 498 499
Data variables:
    Intercept  (chain, draw) float64 32kB 0.04421 0.1077 ... 0.0259 0.06753
    x          (chain, draw) float64 32kB 0.1353 0.232 0.5141 ... 0.2195 0.5014
    y_sigma    (chain, draw) float64 32kB 0.9443 0.9102 0.922 ... 0.9597 0.9249
Attributes:
    created_at:                  2024-04-13T05:34:49.761913+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-dba9a0ee-d95a-4b87-bedc-66a555c05a3d' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-dba9a0ee-d95a-4b87-bedc-66a555c05a3d' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 8</li><li><span class='xr-has-index'>draw</span>: 500</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6008f833-0916-463d-9e58-a1b2fba9da73' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6008f833-0916-463d-9e58-a1b2fba9da73' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-cacdfb00-cb2e-40c8-9cff-3b66cfb5d83c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cacdfb00-cb2e-40c8-9cff-3b66cfb5d83c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-be043bc0-dfb5-4617-8e0e-47c6967a39f1' class='xr-var-data-in' type='checkbox'><label for='data-be043bc0-dfb5-4617-8e0e-47c6967a39f1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 495 496 497 498 499</div><input id='attrs-de0f3a7f-e522-466c-a893-3943bc4e1c04' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de0f3a7f-e522-466c-a893-3943bc4e1c04' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a7b8700e-f1ce-40b3-b53f-2c74264393b1' class='xr-var-data-in' type='checkbox'><label for='data-a7b8700e-f1ce-40b3-b53f-2c74264393b1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 497, 498, 499])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b2d82662-f9aa-4f02-a313-bcedaec1cdae' class='xr-section-summary-in' type='checkbox'  checked><label for='section-b2d82662-f9aa-4f02-a313-bcedaec1cdae' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>Intercept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.04421 0.1077 ... 0.0259 0.06753</div><input id='attrs-1cb2544f-82fa-4e85-9a73-ab8771dc1748' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1cb2544f-82fa-4e85-9a73-ab8771dc1748' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-00880e3f-8057-4124-99f4-12e89ec2d8a9' class='xr-var-data-in' type='checkbox'><label for='data-00880e3f-8057-4124-99f4-12e89ec2d8a9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.04420881,  0.10774889, -0.01477631, ...,  0.05546068,
        -0.01057083,  0.09897323],
       [ 0.00694954,  0.10744512, -0.017276  , ...,  0.19618115,
         0.06402486, -0.01106827],
       [ 0.16110577, -0.07458938,  0.04475104, ...,  0.16745381,
        -0.00406837,  0.07311051],
       ...,
       [ 0.09943931, -0.03684845,  0.09735818, ..., -0.06556524,
         0.11011645,  0.08414361],
       [-0.07703379,  0.02738655,  0.02285994, ...,  0.14379745,
        -0.10339471, -0.02836366],
       [ 0.04903997, -0.03220716, -0.02720002, ...,  0.17203999,
         0.02589751,  0.06752773]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1353 0.232 ... 0.2195 0.5014</div><input id='attrs-77a8a7c7-c108-49dc-b4db-0b58b68cc41f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-77a8a7c7-c108-49dc-b4db-0b58b68cc41f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bb652678-0de6-4bbc-987d-f37b24f7c50b' class='xr-var-data-in' type='checkbox'><label for='data-bb652678-0de6-4bbc-987d-f37b24f7c50b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.13526029, 0.23196226, 0.51413147, ..., 0.23278954, 0.32745043,
        0.37862773],
       [0.40584773, 0.51513052, 0.2268538 , ..., 0.41687492, 0.30601076,
        0.2634667 ],
       [0.41543724, 0.44571834, 0.23530532, ..., 0.6172463 , 0.29822452,
        0.45765768],
       ...,
       [0.49946851, 0.29694244, 0.44142996, ..., 0.26425056, 0.46471836,
        0.32217591],
       [0.41877449, 0.33327679, 0.4045056 , ..., 0.66448843, 0.24280931,
        0.50115044],
       [0.51180277, 0.42393989, 0.56394504, ..., 0.29234944, 0.21949889,
        0.5013853 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y_sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9443 0.9102 ... 0.9597 0.9249</div><input id='attrs-a490d07d-5146-4a33-b986-15d90bf80fc6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a490d07d-5146-4a33-b986-15d90bf80fc6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d6744c6c-9c07-4784-958d-7583c19e2a83' class='xr-var-data-in' type='checkbox'><label for='data-d6744c6c-9c07-4784-958d-7583c19e2a83' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.94428889, 0.91016104, 0.92196855, ..., 0.83634906, 0.79627853,
        1.08163408],
       [0.87025311, 0.85044922, 0.91347637, ..., 1.0028945 , 0.77749843,
        0.87518191],
       [0.94615571, 0.84280628, 1.05011189, ..., 1.0255364 , 0.96478417,
        0.9140493 ],
       ...,
       [0.87146472, 1.04641364, 0.86900166, ..., 0.91303204, 0.95041789,
        0.96797332],
       [0.94906021, 0.99194229, 0.84058257, ..., 0.99087914, 0.96639345,
        0.99059172],
       [0.91025793, 0.8993632 , 1.03222263, ..., 0.9717563 , 0.95967178,
        0.92491709]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6f78dc57-8c1e-4253-9c57-a8785ac8a62e' class='xr-section-summary-in' type='checkbox'  ><label for='section-6f78dc57-8c1e-4253-9c57-a8785ac8a62e' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-7819ea11-85a8-4821-a47f-9667ac60b2c3' class='xr-index-data-in' type='checkbox'/><label for='index-7819ea11-85a8-4821-a47f-9667ac60b2c3' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-0d7fb710-2c7c-4be6-a5d5-f124a40b6e19' class='xr-index-data-in' type='checkbox'/><label for='index-0d7fb710-2c7c-4be6-a5d5-f124a40b6e19' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       490, 491, 492, 493, 494, 495, 496, 497, 498, 499],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=500))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9fd783dc-4871-4155-8977-c37061bc68d9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9fd783dc-4871-4155-8977-c37061bc68d9' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:34:49.761913+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

            <li class = "xr-section-item">
                  <input id="idata_sample_stats29192ab6-43a2-4a9f-81e0-2c211bf8f462" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_stats29192ab6-43a2-4a9f-81e0-2c211bf8f462" class = "xr-section-summary">sample_stats</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 200kB
Dimensions:          (chain: 8, draw: 500)
Coordinates:
  * chain            (chain) int64 64B 0 1 2 3 4 5 6 7
  * draw             (draw) int64 4kB 0 1 2 3 4 5 6 ... 494 495 496 497 498 499
Data variables:
    acceptance_rate  (chain, draw) float64 32kB 0.8137 1.0 1.0 ... 0.9094 0.9834
    diverging        (chain, draw) bool 4kB False False False ... False False
    energy           (chain, draw) float64 32kB 142.0 142.5 ... 143.0 141.5
    lp               (chain, draw) float64 32kB -141.8 -140.8 ... -140.3 -140.3
    n_steps          (chain, draw) int64 32kB 3 3 7 7 7 3 3 7 ... 7 3 7 5 7 3 7
    step_size        (chain, draw) float64 32kB 0.7326 0.7326 ... 0.7643 0.7643
    tree_depth       (chain, draw) int64 32kB 2 2 3 3 3 2 2 3 ... 3 2 3 3 3 2 3
Attributes:
    created_at:                  2024-04-13T05:34:49.763427+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-6cd86f5c-f7b5-42fd-b4a8-85d61c804b79' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6cd86f5c-f7b5-42fd-b4a8-85d61c804b79' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 8</li><li><span class='xr-has-index'>draw</span>: 500</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-0a853fdf-921c-4973-9dff-3931e2160344' class='xr-section-summary-in' type='checkbox'  checked><label for='section-0a853fdf-921c-4973-9dff-3931e2160344' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-cfc5327e-feed-4fb6-9ffc-58ed6cd1492e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cfc5327e-feed-4fb6-9ffc-58ed6cd1492e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9a3f040c-02bc-42e0-ac7f-9c27d8fc3189' class='xr-var-data-in' type='checkbox'><label for='data-9a3f040c-02bc-42e0-ac7f-9c27d8fc3189' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 495 496 497 498 499</div><input id='attrs-a5654549-c4e8-4b1a-aed1-bae53d1206ad' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a5654549-c4e8-4b1a-aed1-bae53d1206ad' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-24b55eac-10c3-45a9-a77d-6b1c2736b3c0' class='xr-var-data-in' type='checkbox'><label for='data-24b55eac-10c3-45a9-a77d-6b1c2736b3c0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 497, 498, 499])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6ae28e77-68e9-4cf6-b351-67dec5408deb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6ae28e77-68e9-4cf6-b351-67dec5408deb' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>acceptance_rate</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8137 1.0 1.0 ... 0.9094 0.9834</div><input id='attrs-d40afe1b-9359-4a1e-8e91-56e00d590b40' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d40afe1b-9359-4a1e-8e91-56e00d590b40' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-710d41b7-c486-4644-bca1-470de66a2a70' class='xr-var-data-in' type='checkbox'><label for='data-710d41b7-c486-4644-bca1-470de66a2a70' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.81368466, 1.        , 1.        , ..., 0.95726715, 0.95332204,
        1.        ],
       [0.98623155, 0.76947694, 1.        , ..., 0.6501189 , 0.6980205 ,
        1.        ],
       [0.98539058, 0.82802334, 0.96559601, ..., 0.72850635, 1.        ,
        0.86563511],
       ...,
       [0.79238494, 0.97989654, 0.94005541, ..., 0.98283263, 0.99321313,
        0.92314755],
       [0.95823733, 0.94198648, 0.91853339, ..., 0.68699656, 0.972578  ,
        0.74390253],
       [0.99181102, 0.97429544, 0.78790853, ..., 1.        , 0.90941548,
        0.98341956]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-e8cf1517-8ae0-4fc7-a72b-7b033253577f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e8cf1517-8ae0-4fc7-a72b-7b033253577f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2e504750-e948-484c-8196-fce975a0efa9' class='xr-var-data-in' type='checkbox'><label for='data-2e504750-e948-484c-8196-fce975a0efa9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>142.0 142.5 141.8 ... 143.0 141.5</div><input id='attrs-d600d658-ea52-4da4-abb5-a2c86f0a1dde' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d600d658-ea52-4da4-abb5-a2c86f0a1dde' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f47e2692-0ad4-44a6-873a-a4dace06f63b' class='xr-var-data-in' type='checkbox'><label for='data-f47e2692-0ad4-44a6-873a-a4dace06f63b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[141.967537  , 142.54106626, 141.78398912, ..., 142.07442022,
        143.72755872, 142.21633731],
       [142.4254147 , 142.09857076, 141.94644772, ..., 142.39410249,
        146.51261269, 143.65723574],
       [141.4414682 , 142.96701481, 142.39395521, ..., 144.02150538,
        142.66721068, 140.85225446],
       ...,
       [142.45649528, 142.75638401, 142.59874467, ..., 141.35824722,
        140.94450824, 141.2808887 ],
       [140.3251331 , 141.16320788, 140.88902952, ..., 144.52035375,
        144.09031991, 143.92351871],
       [142.65283365, 141.01504212, 142.60582761, ..., 143.46419056,
        142.97607812, 141.46662296]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lp</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-141.8 -140.8 ... -140.3 -140.3</div><input id='attrs-dcc6c4c8-c798-4d5f-ad4b-4094ff64e04b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dcc6c4c8-c798-4d5f-ad4b-4094ff64e04b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-22e2d080-fb39-48df-b1f8-2addbf3d222a' class='xr-var-data-in' type='checkbox'><label for='data-22e2d080-fb39-48df-b1f8-2addbf3d222a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-141.78590569, -140.75355135, -140.61320047, ..., -141.63502666,
        -142.12600187, -141.57227603],
       [-139.88014501, -141.79255751, -140.226333  , ..., -141.301519  ,
        -143.3329595 , -140.20584575],
       [-140.39038429, -141.56925705, -141.27509741, ..., -143.36048355,
        -139.58615368, -139.84922801],
       ...,
       [-140.99893216, -140.82540718, -140.38825538, ..., -140.12098164,
        -140.10850196, -139.71074945],
       [-140.09932106, -139.69086444, -140.49414807, ..., -143.90263595,
        -140.69641315, -140.7183776 ],
       [-140.46042516, -139.8366111 , -142.15416918, ..., -140.96117584,
        -140.27772734, -140.27024162]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>n_steps</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>3 3 7 7 7 3 3 7 ... 7 7 3 7 5 7 3 7</div><input id='attrs-d3232e19-a4d5-42ba-8885-6de69a893062' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d3232e19-a4d5-42ba-8885-6de69a893062' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-06c8956a-69e2-49da-9073-393608284373' class='xr-var-data-in' type='checkbox'><label for='data-06c8956a-69e2-49da-9073-393608284373' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3,  3,  7, ...,  7,  7,  7],
       [ 3,  3,  3, ...,  3,  7,  3],
       [11,  3,  3, ...,  3,  3,  3],
       ...,
       [ 7,  7,  7, ...,  7,  7,  3],
       [ 3,  3,  3, ...,  7,  7,  3],
       [ 7,  3,  3, ...,  7,  3,  7]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.7326 0.7326 ... 0.7643 0.7643</div><input id='attrs-f852f763-1819-42a7-8151-b8b06089576f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f852f763-1819-42a7-8151-b8b06089576f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-95d28e08-e955-4a1c-aa52-75713555c005' class='xr-var-data-in' type='checkbox'><label for='data-95d28e08-e955-4a1c-aa52-75713555c005' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.73264667, 0.73264667, 0.73264667, ..., 0.73264667, 0.73264667,
        0.73264667],
       [0.84139296, 0.84139296, 0.84139296, ..., 0.84139296, 0.84139296,
        0.84139296],
       [0.90832794, 0.90832794, 0.90832794, ..., 0.90832794, 0.90832794,
        0.90832794],
       ...,
       [0.75868138, 0.75868138, 0.75868138, ..., 0.75868138, 0.75868138,
        0.75868138],
       [0.83356209, 0.83356209, 0.83356209, ..., 0.83356209, 0.83356209,
        0.83356209],
       [0.76429536, 0.76429536, 0.76429536, ..., 0.76429536, 0.76429536,
        0.76429536]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tree_depth</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>2 2 3 3 3 2 2 3 ... 3 3 2 3 3 3 2 3</div><input id='attrs-c4103393-57b5-4ed7-9d44-715198271877' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c4103393-57b5-4ed7-9d44-715198271877' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-82d8c848-98ec-427c-a803-b4610e14685e' class='xr-var-data-in' type='checkbox'><label for='data-82d8c848-98ec-427c-a803-b4610e14685e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2, 2, 3, ..., 3, 3, 3],
       [2, 2, 2, ..., 2, 3, 2],
       [4, 2, 2, ..., 2, 2, 2],
       ...,
       [3, 3, 3, ..., 3, 3, 2],
       [2, 2, 2, ..., 3, 3, 2],
       [3, 2, 2, ..., 3, 2, 3]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2aff4cba-e1b7-43ad-b4d3-2bf34be52e2e' class='xr-section-summary-in' type='checkbox'  ><label for='section-2aff4cba-e1b7-43ad-b4d3-2bf34be52e2e' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-59c99134-8157-45c2-9cb7-6aa54dea8994' class='xr-index-data-in' type='checkbox'/><label for='index-59c99134-8157-45c2-9cb7-6aa54dea8994' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-c5f3c593-b980-4d7c-9b01-55c7fb725918' class='xr-index-data-in' type='checkbox'/><label for='index-c5f3c593-b980-4d7c-9b01-55c7fb725918' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       490, 491, 492, 493, 494, 495, 496, 497, 498, 499],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=500))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8f8d89fc-e7f0-4f42-98e5-61aa2e35198d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8f8d89fc-e7f0-4f42-98e5-61aa2e35198d' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:34:49.763427+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-sections.group-sections {
  grid-template-columns: auto;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
.xr-wrap{width:700px!important;} </style>



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


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
</pre>







            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">

            <li class = "xr-section-item">
                  <input id="idata_posteriorffbc0821-a9ed-442a-8ab6-863a47f4820f" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posteriorffbc0821-a9ed-442a-8ab6-863a47f4820f" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 26kB
Dimensions:    (chain: 4, draw: 250)
Coordinates:
  * chain      (chain) int64 32B 0 1 2 3
  * draw       (draw) int64 2kB 0 1 2 3 4 5 6 7 ... 243 244 245 246 247 248 249
Data variables:
    Intercept  (chain, draw) float64 8kB 0.112 -0.08016 ... -0.04784 -0.04427
    x          (chain, draw) float64 8kB 0.4311 0.3077 0.2568 ... 0.557 0.4814
    y_sigma    (chain, draw) float64 8kB 0.9461 0.9216 0.9021 ... 0.9574 0.9414
Attributes:
    created_at:                  2024-04-13T05:36:20.439151+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-121d6ffa-fa06-4446-99a9-2148b02197d7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-121d6ffa-fa06-4446-99a9-2148b02197d7' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 250</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-45be4ede-08de-4ddd-8b87-0e74159feaf1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-45be4ede-08de-4ddd-8b87-0e74159feaf1' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-7bb2b488-f21f-43b2-975d-6abac2555516' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7bb2b488-f21f-43b2-975d-6abac2555516' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-707c5d97-e7d8-48a8-ad12-b62284b2a390' class='xr-var-data-in' type='checkbox'><label for='data-707c5d97-e7d8-48a8-ad12-b62284b2a390' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 245 246 247 248 249</div><input id='attrs-ac02668d-0e69-415c-85c0-6d38bacde98d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ac02668d-0e69-415c-85c0-6d38bacde98d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bc1dc0e0-8f02-4956-999b-2d235fccc9e5' class='xr-var-data-in' type='checkbox'><label for='data-bc1dc0e0-8f02-4956-999b-2d235fccc9e5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 247, 248, 249])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e33d53b0-5d8c-4028-abc6-8f16ce2216ba' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e33d53b0-5d8c-4028-abc6-8f16ce2216ba' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>Intercept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.112 -0.08016 ... -0.04427</div><input id='attrs-0b15409f-24e5-452a-9b2d-f33a37d7826a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0b15409f-24e5-452a-9b2d-f33a37d7826a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a15fa394-ab36-4618-898e-cd47e54705ec' class='xr-var-data-in' type='checkbox'><label for='data-a15fa394-ab36-4618-898e-cd47e54705ec' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.12014970e-01, -8.01554356e-02, -5.13623666e-02,
         3.14326884e-02, -1.15157203e-02,  8.29955919e-02,
         1.93598964e-02,  3.52348779e-02,  1.16428197e-01,
         8.72568457e-02,  8.72568457e-02, -2.82423636e-02,
        -1.84281659e-01, -7.19613600e-02,  1.91168564e-01,
        -1.23364451e-01, -1.19337859e-01, -3.08974539e-02,
        -7.18298964e-02,  1.19861654e-01, -1.00735526e-01,
         6.32700448e-02, -5.72727355e-03,  1.17338655e-01,
        -3.83727222e-02,  1.84746578e-01,  7.26989694e-02,
         1.94564654e-01,  1.44827239e-01, -8.50718497e-02,
         1.87666705e-01, -1.60104555e-01, -4.99203302e-02,
         3.19979141e-03, -3.94543945e-03,  8.66672888e-04,
         4.33390733e-02,  5.22653528e-02, -4.77497368e-02,
        -1.88745071e-02,  5.03627424e-02,  8.24434767e-02,
        -3.79889140e-03,  7.70856139e-03, -4.77259521e-02,
        -1.90736318e-02,  6.76733158e-02,  5.14461069e-02,
        -4.56113715e-02,  1.60543248e-01,  7.32836299e-02,
         2.28842579e-03,  3.05194139e-02,  4.27895103e-02,
        -2.51634507e-02, -4.60161935e-02,  2.44964388e-01,
         2.76318153e-01,  4.33818171e-02,  1.46208904e-01,
...
        -5.68838115e-02, -6.14275201e-02, -5.96425618e-02,
         6.11356758e-02,  6.42661723e-03,  5.41912583e-02,
        -1.76976244e-01, -4.62930404e-02,  9.61963932e-02,
        -1.40433636e-01,  2.05056910e-01,  1.82385197e-01,
         1.21005125e-01, -9.65523825e-02,  9.11450646e-02,
         1.49525640e-02, -8.32289763e-02,  3.24479331e-02,
         1.09007071e-02, -6.92830705e-02,  6.64926592e-02,
         3.23060974e-02, -1.73437807e-01, -1.25619389e-03,
         8.89183729e-02,  1.02309051e-01,  4.12736086e-02,
         1.03893380e-01,  6.89267255e-02,  1.37649597e-01,
        -7.63849028e-02,  7.69987215e-02, -1.14605433e-01,
        -1.59066163e-01,  2.02049201e-01,  1.67222994e-01,
        -5.02468032e-02, -1.17601875e-01, -1.67595598e-03,
        -1.97669449e-01,  4.36079372e-02,  8.41929183e-02,
         1.31836071e-01,  1.65427331e-01,  1.26585460e-01,
        -7.27516393e-02,  5.41849189e-02,  2.21844869e-02,
         5.94315594e-02,  5.94315594e-02,  7.45566691e-02,
        -9.33357688e-03, -4.93686976e-02, -3.43353187e-02,
         6.89221401e-02, -3.19652375e-02, -4.78438315e-02,
        -4.42738699e-02]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.4311 0.3077 ... 0.557 0.4814</div><input id='attrs-4d1d95c1-71df-4ee6-a8e1-e583d3510d9b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4d1d95c1-71df-4ee6-a8e1-e583d3510d9b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d3df1e5f-5c71-415d-b6cd-310774e81b89' class='xr-var-data-in' type='checkbox'><label for='data-d3df1e5f-5c71-415d-b6cd-310774e81b89' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.43113703, 0.30774729, 0.25682166, 0.31641395, 0.51546729,
        0.31100976, 0.44043531, 0.38164497, 0.42080014, 0.30069041,
        0.30069041, 0.53305267, 0.11871418, 0.22919114, 0.22043383,
        0.32368544, 0.28827739, 0.44216387, 0.38292596, 0.38328387,
        0.31277052, 0.28380182, 0.39125062, 0.5436668 , 0.19914823,
        0.23381157, 0.3952613 , 0.48281672, 0.27598205, 0.46597795,
        0.48635971, 0.21363092, 0.39350997, 0.42601567, 0.3035345 ,
        0.26553072, 0.44019149, 0.34397815, 0.23609522, 0.53683168,
        0.45841485, 0.23891478, 0.54442998, 0.16697332, 0.19146859,
        0.22799538, 0.39366724, 0.37134365, 0.34501806, 0.37506017,
        0.28311981, 0.16254121, 0.61289656, 0.13063232, 0.03017502,
        0.18434623, 0.36065819, 0.52235008, 0.24458848, 0.14313226,
        0.22279879, 0.44892021, 0.3952106 , 0.34290512, 0.42439318,
        0.23102895, 0.19110882, 0.25093658, 0.37681057, 0.36135287,
        0.30745033, 0.27562781, 0.31724922, 0.45716849, 0.47116505,
        0.43884602, 0.43553571, 0.29161261, 0.41998198, 0.40796597,
        0.30405689, 0.31259796, 0.20570747, 0.39392466, 0.31348596,
        0.4214938 , 0.52463068, 0.16792862, 0.28029374, 0.16153929,
        0.16724633, 0.08144633, 0.19192458, 0.34938819, 0.13305379,
        0.13881198, 0.37849938, 0.37084368, 0.27404992, 0.46209003,
...
        0.36307521, 0.28227501, 0.38551525, 0.23261809, 0.36514994,
        0.35783934, 0.3823261 , 0.30670976, 0.32886498, 0.37068029,
        0.34013729, 0.52474148, 0.639228  , 0.09371894, 0.42023135,
        0.36733482, 0.40599032, 0.24382963, 0.40221825, 0.3047073 ,
        0.24407962, 0.30213837, 0.44363912, 0.57883031, 0.48781764,
        0.48525391, 0.29198732, 0.37745175, 0.31777746, 0.28262044,
        0.18702892, 0.51867304, 0.52340339, 0.24125419, 0.23332597,
        0.46851727, 0.46079104, 0.32301517, 0.35635714, 0.33111389,
        0.37437903, 0.28657985, 0.43734974, 0.35478284, 0.30887643,
        0.49867288, 0.2525673 , 0.33079942, 0.02800324, 0.31465776,
        0.4585882 , 0.28368126, 0.4896697 , 0.44762422, 0.41453835,
        0.246885  , 0.37482138, 0.40059614, 0.27591068, 0.21900013,
        0.47128275, 0.21132567, 0.39900367, 0.2329504 , 0.39579287,
        0.37344961, 0.34516518, 0.32227915, 0.35271413, 0.37687565,
        0.31151605, 0.37301695, 0.26012957, 0.30024098, 0.2745939 ,
        0.25698144, 0.37064686, 0.43608796, 0.29833848, 0.4057974 ,
        0.37998817, 0.3505483 , 0.3385325 , 0.29122156, 0.46273061,
        0.26565498, 0.27156778, 0.37355743, 0.30409736, 0.34103447,
        0.33149781, 0.33149781, 0.43853323, 0.27834059, 0.36111445,
        0.40162141, 0.42356887, 0.55111442, 0.55695307, 0.48135813]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y_sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9461 0.9216 ... 0.9574 0.9414</div><input id='attrs-bafe04c0-421a-47c1-9151-420c27e95e40' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bafe04c0-421a-47c1-9151-420c27e95e40' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-93e2ae7a-7897-4503-af6e-b1a87169a1bc' class='xr-var-data-in' type='checkbox'><label for='data-93e2ae7a-7897-4503-af6e-b1a87169a1bc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.94608021, 0.92160426, 0.90211804, 0.8784906 , 1.00443151,
        0.96727798, 0.87524178, 0.96817024, 0.86457884, 1.0262922 ,
        1.0262922 , 0.84391933, 0.98129848, 1.00988048, 0.99554283,
        0.88587989, 1.00586764, 0.92997466, 0.94168997, 0.99374899,
        0.92810392, 0.96878376, 0.90682728, 0.94883541, 1.01957489,
        1.02278733, 0.90160336, 0.94497765, 0.87888132, 0.92205875,
        0.9138956 , 0.96519328, 1.06316311, 0.84402636, 0.83729644,
        0.89811997, 0.97144791, 0.98208145, 0.91289233, 0.96673035,
        0.95542624, 0.91245841, 0.96527727, 0.92783747, 1.03786087,
        0.94764661, 1.00547045, 0.85588467, 0.98223118, 0.8674327 ,
        0.94037555, 0.91725845, 0.99391199, 0.92434293, 0.9638643 ,
        1.08815478, 1.01399545, 1.02349856, 0.92934388, 0.96598116,
        0.96311436, 0.93945143, 0.89124759, 0.98455184, 0.89591612,
        1.006701  , 0.95597051, 1.00027136, 0.91409196, 0.97378494,
        0.87137146, 0.87160277, 1.04749666, 0.8805835 , 0.8819731 ,
        0.88645983, 1.00263402, 0.88708112, 0.99995189, 1.01743406,
        0.87473936, 0.9076109 , 1.02202715, 0.88250374, 1.06665137,
        0.84538309, 0.84109731, 1.0524254 , 0.97522117, 0.94564838,
        1.05965236, 0.97217503, 0.96459187, 0.9413301 , 1.00422163,
        1.00733854, 0.95474848, 0.94441562, 0.89236671, 0.96775448,
...
        0.9959878 , 0.93569281, 0.96401675, 0.88786078, 1.14540889,
        0.8224594 , 0.84935646, 1.03698789, 1.00625543, 0.88735547,
        0.99331278, 1.00797432, 0.94295773, 0.98513086, 1.0195952 ,
        0.88995881, 0.84278984, 1.02888997, 0.96128787, 0.91245996,
        0.97871983, 0.89146682, 0.98259937, 0.95369473, 1.02821356,
        0.83242344, 1.06338194, 0.82728423, 1.06433136, 0.85249613,
        0.92553966, 0.96450458, 1.05280513, 0.90353168, 0.84823849,
        1.03949674, 0.92214448, 0.9231072 , 0.87897527, 0.95304901,
        0.91455056, 0.97220005, 0.91253068, 0.92932491, 0.85741327,
        1.05336522, 1.05774423, 1.20149457, 0.99443219, 0.89727566,
        0.97462237, 0.9137672 , 0.99391023, 0.98467151, 0.83221799,
        1.06702143, 0.85499338, 0.94884501, 0.94337727, 0.96101538,
        0.87323245, 1.02556183, 0.95388553, 0.97263382, 0.98591673,
        0.96502309, 0.85746496, 0.84968585, 0.99422795, 0.89441428,
        1.04297339, 1.05277335, 0.85709214, 0.87885518, 1.03047245,
        1.05704007, 0.91158198, 0.91662192, 0.90469643, 0.96868723,
        0.91674164, 0.93328151, 0.91403954, 1.22344839, 0.98301442,
        0.97414525, 0.97886513, 0.91856841, 0.95869794, 0.92022874,
        0.92707182, 0.92707182, 0.95625616, 0.93941143, 0.93244475,
        0.93337728, 0.97659533, 0.97438746, 0.95742502, 0.94139699]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-71b15a55-fe0d-4b59-9425-614e1a906f90' class='xr-section-summary-in' type='checkbox'  ><label for='section-71b15a55-fe0d-4b59-9425-614e1a906f90' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-911e85a3-281e-422a-ba50-2fa5fae51711' class='xr-index-data-in' type='checkbox'/><label for='index-911e85a3-281e-422a-ba50-2fa5fae51711' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-d8f27767-cb2a-4c2f-99a3-8b5453f106d8' class='xr-index-data-in' type='checkbox'/><label for='index-d8f27767-cb2a-4c2f-99a3-8b5453f106d8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       240, 241, 242, 243, 244, 245, 246, 247, 248, 249],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=250))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e41a2348-0369-4c02-9ac6-796a24921225' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e41a2348-0369-4c02-9ac6-796a24921225' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:36:20.439151+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

            <li class = "xr-section-item">
                  <input id="idata_sample_statse481c5a2-d674-4367-a9d5-2ce40249f625" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_statse481c5a2-d674-4367-a9d5-2ce40249f625" class = "xr-section-summary">sample_stats</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 51kB
Dimensions:          (chain: 4, draw: 250)
Coordinates:
  * chain            (chain) int64 32B 0 1 2 3
  * draw             (draw) int64 2kB 0 1 2 3 4 5 6 ... 244 245 246 247 248 249
Data variables:
    acceptance_rate  (chain, draw) float64 8kB 0.8489 0.9674 ... 0.983 1.0
    diverging        (chain, draw) bool 1kB False False False ... False False
    energy           (chain, draw) float64 8kB 141.2 140.7 140.5 ... 141.9 141.4
    lp               (chain, draw) float64 8kB -139.9 -140.0 ... -141.6 -140.3
    n_steps          (chain, draw) int64 8kB 3 7 7 3 3 3 7 3 ... 3 3 3 3 3 3 3 3
    step_size        (chain, draw) float64 8kB 0.8923 0.8923 ... 0.9726 0.9726
    tree_depth       (chain, draw) int64 8kB 2 3 3 2 2 2 3 2 ... 2 2 2 2 2 2 2 2
Attributes:
    created_at:                  2024-04-13T05:36:20.441267+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-2b5c9424-e2f8-4a02-b97d-d04cc7108e98' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-2b5c9424-e2f8-4a02-b97d-d04cc7108e98' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 250</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-bcc12a20-5750-46aa-b6b2-546675a1c0bf' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bcc12a20-5750-46aa-b6b2-546675a1c0bf' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-1c9df9c7-c528-4aa5-b52a-c3e05594df6f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c9df9c7-c528-4aa5-b52a-c3e05594df6f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2acc25de-8742-4eab-94fd-a18cfec41b8d' class='xr-var-data-in' type='checkbox'><label for='data-2acc25de-8742-4eab-94fd-a18cfec41b8d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 245 246 247 248 249</div><input id='attrs-dcb640f8-4dac-49ee-92fa-d9c0bfec9e03' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dcb640f8-4dac-49ee-92fa-d9c0bfec9e03' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6569833b-9a66-41d9-9c6d-72f0aad19a89' class='xr-var-data-in' type='checkbox'><label for='data-6569833b-9a66-41d9-9c6d-72f0aad19a89' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 247, 248, 249])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6587e0d8-83eb-4fe3-9a36-a8b9c1128045' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6587e0d8-83eb-4fe3-9a36-a8b9c1128045' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>acceptance_rate</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8489 0.9674 0.9733 ... 0.983 1.0</div><input id='attrs-e49fae2e-3546-4b9f-9ab4-25f76a8f9bc2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e49fae2e-3546-4b9f-9ab4-25f76a8f9bc2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca534e62-a059-4b4e-86c3-944b11540616' class='xr-var-data-in' type='checkbox'><label for='data-ca534e62-a059-4b4e-86c3-944b11540616' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.84888924, 0.9673831 , 0.9733441 , 0.89329671, 0.86130558,
        0.70132658, 0.96381014, 1.        , 0.92972425, 0.99393407,
        0.42991297, 0.91174018, 0.87340076, 1.        , 0.9589841 ,
        0.87037497, 1.        , 1.        , 0.99496101, 0.88726023,
        0.9885737 , 0.90697071, 0.99084364, 0.80724362, 0.96962189,
        0.9833609 , 0.97786349, 0.86134776, 0.90956489, 0.90854708,
        0.98462356, 0.89669834, 0.90547798, 0.98699424, 0.97484471,
        1.        , 0.88021684, 0.97615242, 0.96073465, 0.91237892,
        0.9845141 , 0.94372409, 0.92777163, 0.97342742, 0.95057722,
        1.        , 0.9612599 , 0.98843436, 1.        , 0.8602412 ,
        0.99473406, 0.82449416, 0.87100299, 0.89980582, 0.91056175,
        1.        , 0.9853551 , 0.6339886 , 0.79539193, 0.79552437,
        0.98590884, 0.7882424 , 0.95636624, 1.        , 0.91644093,
        0.61071389, 0.9079233 , 1.        , 0.94421384, 0.89397927,
        0.97441365, 0.74609534, 0.87623779, 0.65668758, 1.        ,
        0.99754243, 1.        , 1.        , 0.91660863, 0.99472955,
        0.7589626 , 1.        , 0.79027932, 0.96149457, 0.86777121,
        0.98246178, 0.78602939, 0.98883522, 0.99371046, 0.87781383,
        0.90548202, 0.91736797, 0.93188359, 1.        , 0.56722994,
        0.92168357, 1.        , 0.99974158, 0.85855488, 0.9130902 ,
...
        0.933183  , 0.9759323 , 0.80154439, 0.58682022, 0.89919977,
        0.96192085, 1.        , 1.        , 1.        , 0.92846989,
        0.73777125, 0.78583409, 0.84084899, 0.99937813, 0.68385273,
        0.90734862, 0.87463518, 0.86085163, 0.96549202, 1.        ,
        0.73037943, 0.94028211, 1.        , 0.82836097, 0.98809743,
        0.81764963, 1.        , 1.        , 0.98798519, 0.98349015,
        0.67753562, 0.95280466, 0.90283301, 0.93024677, 0.93151669,
        1.        , 0.96818253, 0.99781367, 0.91713181, 0.96014857,
        1.        , 0.75023019, 0.96045113, 0.94362692, 0.7532761 ,
        0.83880141, 0.87182051, 0.74171229, 0.94450692, 1.        ,
        0.94232357, 0.86828235, 0.92197311, 1.        , 0.68644282,
        0.97359373, 0.90638899, 1.        , 0.76594222, 0.91762793,
        0.95915589, 0.87177042, 1.        , 0.82258746, 0.81200426,
        1.        , 0.93555039, 0.8848828 , 1.        , 0.87053966,
        0.85639141, 0.97637026, 0.98528981, 1.        , 0.85350081,
        0.95414825, 1.        , 1.        , 0.98446748, 0.80153679,
        1.        , 0.62828104, 1.        , 0.36846189, 0.86386322,
        0.64099386, 1.        , 0.84170202, 0.95361984, 0.9205592 ,
        0.98298569, 0.66323733, 0.85359733, 0.98170805, 0.99093396,
        0.97552769, 0.95590249, 0.84734008, 0.98296291, 1.        ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-c54d7a15-c912-49af-a677-212c49805b0d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c54d7a15-c912-49af-a677-212c49805b0d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca533660-c33b-47fd-b775-81e15ebaa711' class='xr-var-data-in' type='checkbox'><label for='data-ca533660-c33b-47fd-b775-81e15ebaa711' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
...
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>141.2 140.7 140.5 ... 141.9 141.4</div><input id='attrs-9412bd1e-194c-4fc2-b956-97adcbfaac74' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9412bd1e-194c-4fc2-b956-97adcbfaac74' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0b101589-f328-41a1-be73-eb77897fe552' class='xr-var-data-in' type='checkbox'><label for='data-0b101589-f328-41a1-be73-eb77897fe552' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[141.20161744, 140.70598666, 140.49148926, 141.20239103,
        141.19010811, 144.18660929, 140.33281704, 139.96308322,
        140.57893835, 141.38941515, 148.64059094, 142.93289825,
        146.04299163, 144.14605027, 142.67642036, 144.14576568,
        141.7660787 , 141.06947956, 140.19356762, 141.08246659,
        141.00591894, 141.35859185, 139.93979496, 141.51824106,
        142.56997862, 142.56991682, 142.58758086, 141.72338033,
        142.8969296 , 142.42932381, 141.68924271, 143.93789916,
        143.12463584, 142.28934616, 141.66880823, 140.75972524,
        141.01378042, 140.13124997, 140.51835247, 141.61498126,
        141.31832015, 140.66043205, 142.12540545, 141.64700326,
        142.85807033, 141.70816164, 140.81111539, 140.31405788,
        140.29502047, 142.17134901, 141.34942234, 141.96753663,
        144.45306366, 144.30037464, 144.92560947, 145.88202322,
        145.57638211, 149.53691678, 146.82447002, 143.92894787,
        142.945291  , 143.62675737, 140.2703418 , 140.10974452,
        140.59959866, 143.312126  , 143.4798215 , 142.32917667,
        141.59167508, 140.91590845, 141.55742637, 142.42319463,
        142.17191796, 145.8462414 , 143.50482308, 143.38838779,
        140.93972277, 140.42443743, 141.12402307, 141.13791   ,
...
        141.93825369, 144.56393892, 142.73608027, 144.07625796,
        142.0531902 , 141.41700837, 141.51465137, 142.62280631,
        143.79817982, 142.76116047, 143.85988869, 142.11378708,
        141.76697822, 141.60731201, 141.3540055 , 139.81189634,
        140.0125599 , 140.24700481, 139.84972186, 141.52789778,
        141.49476174, 140.70968672, 141.66856485, 142.71296198,
        143.85619075, 145.52274885, 149.21963527, 147.57321087,
        140.39448216, 141.853325  , 143.43444642, 141.78434647,
        145.09952198, 142.36990061, 143.10356634, 140.37811064,
        141.38857059, 141.31196017, 140.94430523, 142.22846388,
        141.29718806, 141.2077227 , 142.47077178, 141.54519327,
        140.3278173 , 140.95240769, 140.60694954, 140.58587491,
        140.9740406 , 141.73032551, 142.70897439, 141.88098807,
        141.65131353, 142.52300291, 143.27844069, 141.46619993,
        140.87239229, 141.57738767, 140.5790518 , 143.21027453,
        141.86795416, 146.14898606, 147.39883864, 145.13195593,
        141.11515884, 141.95224649, 140.49781241, 140.32029504,
        139.45421819, 142.18814629, 140.68012531, 139.92677613,
        139.90305713, 139.92528181, 140.02423793, 141.83956437,
        141.90308307, 141.4326794 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lp</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-139.9 -140.0 ... -141.6 -140.3</div><input id='attrs-3765ddd2-c511-4d38-95e8-aba6322297ae' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3765ddd2-c511-4d38-95e8-aba6322297ae' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a373c01e-c4dd-4a4b-9c56-211c549a2530' class='xr-var-data-in' type='checkbox'><label for='data-a373c01e-c4dd-4a4b-9c56-211c549a2530' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-139.87893778, -139.9645816 , -140.15023671, -139.72915805,
        -140.95230947, -139.75107499, -140.01457839, -139.4408409 ,
        -140.52869623, -140.55137532, -140.55137532, -142.39342002,
        -143.70774595, -140.91220502, -142.25708851, -140.87749392,
        -140.94220804, -139.82075105, -139.84129544, -140.14606113,
        -140.18660279, -139.81366331, -139.4446184 , -141.12989093,
        -141.18051022, -142.24699337, -139.55294872, -141.34902122,
        -141.1207297 , -140.68010125, -141.41084446, -141.82800678,
        -141.20536354, -140.59228081, -140.69416166, -139.86811985,
        -139.72750674, -139.60666827, -140.26256251, -141.00322146,
        -139.77866637, -140.37414036, -141.03275542, -141.08023336,
        -141.6000328 , -140.14892777, -139.96123244, -140.06522244,
        -139.78981573, -141.01735455, -139.75775472, -141.22535186,
        -142.38556034, -141.97659086, -144.27384526, -142.67574989,
        -142.35722033, -143.75557052, -139.98097631, -142.70952977,
        -140.7217984 , -139.6410683 , -139.84543799, -139.64193129,
        -139.6247126 , -140.62462715, -142.18994991, -140.89554653,
        -139.30901481, -140.66999757, -140.09430283, -140.18511655,
        -141.47845726, -143.08700779, -142.63691111, -140.3748972 ,
        -140.30603842, -139.8354723 , -140.91681837, -140.8016408 ,
...
        -141.61328651, -142.16610216, -140.86932964, -141.66349492,
        -141.24899794, -140.92756748, -141.35783924, -140.46980857,
        -141.82709759, -141.4470513 , -141.67479038, -140.53378286,
        -141.14396647, -140.86380423, -139.8673658 , -139.34853841,
        -139.70693196, -139.74296053, -139.57680571, -140.83672532,
        -140.07834751, -139.67448254, -140.52919726, -141.42318554,
        -141.21657892, -144.92174257, -145.62381945, -139.76482799,
        -140.08610874, -141.02022716, -141.7757176 , -141.0336296 ,
        -141.39137894, -141.95233948, -140.33931036, -139.38387579,
        -140.13741269, -140.30673995, -140.39845601, -141.29817581,
        -139.47208887, -140.20293725, -141.72161449, -139.43675991,
        -140.300563  , -140.72081646, -139.7054349 , -139.82631079,
        -140.69282006, -141.25803608, -141.07708316, -140.04546292,
        -141.29624749, -142.45655409, -141.21055678, -140.6477156 ,
        -139.8087491 , -140.65559625, -139.34928981, -142.04465792,
        -139.35271669, -145.90075   , -140.47978836, -141.11111359,
        -140.53840869, -139.8545253 , -139.57336607, -139.29539907,
        -139.39810176, -139.39810176, -139.70660711, -139.62054515,
        -139.56840864, -139.5697715 , -139.73762165, -141.38313388,
        -141.58458337, -140.34263295]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>n_steps</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>3 7 7 3 3 3 7 3 ... 3 3 3 3 3 3 3 3</div><input id='attrs-32062937-ab60-4227-9890-b15b6124e58e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-32062937-ab60-4227-9890-b15b6124e58e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3fb48da5-626d-44ec-a184-264883283c18' class='xr-var-data-in' type='checkbox'><label for='data-3fb48da5-626d-44ec-a184-264883283c18' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3,  7,  7,  3,  3,  3,  7,  3,  3, 11,  3,  3,  7,  3,  7,  7,
         3,  3,  7,  5,  7,  7,  3,  3,  7,  7,  7,  7,  3,  7,  7,  7,
         7,  3,  7,  3,  3,  3,  3,  7,  7,  3,  3,  7,  7,  3,  7,  3,
         3,  7,  3,  3,  7,  3,  3,  3,  7,  3,  3,  3,  7,  3,  3,  3,
         3,  7,  7,  3,  3,  7,  7,  3,  3,  3,  3,  7,  3,  7,  3,  3,
         3,  1,  7,  7,  3,  3,  7,  3,  3,  7,  3,  3,  3,  3,  3,  3,
         3,  3,  3,  7,  7,  7,  3,  3,  3,  7,  7,  7,  7,  3,  5,  3,
         1,  7,  7,  3,  3,  7,  3,  3, 15,  3,  7,  3,  3,  3,  3,  3,
         3,  3,  7,  7,  7,  3,  3,  7,  7,  3,  3,  3,  7,  7,  3,  3,
         3,  3,  3,  3,  3,  7,  7,  7,  3,  3,  7,  3,  5,  3,  3,  3,
         3,  3,  3,  1,  3,  3,  3,  7,  1,  3,  7,  7,  3,  3,  3,  3,
         7,  3,  1,  3,  1,  3,  3,  7,  3,  3, 15,  3,  3,  7,  3,  3,
         3,  3,  7,  3,  3,  3,  3, 15,  3,  7,  1,  3,  1,  3,  3,  7,
         3,  3,  7,  3,  3,  7,  7,  7,  7,  3,  3,  7,  7,  7,  3,  7,
         3,  7,  3,  3,  7,  3,  3,  3,  7,  3,  3,  7,  3,  3,  7,  3,
         7,  3,  3,  7,  7,  3,  3,  3,  7,  3],
       [ 3,  7,  3,  3,  3,  3,  7,  3,  3,  3,  3,  3,  3,  3,  3,  3,
         3,  3,  7,  3,  3, 11,  3,  3,  3,  3,  3,  7,  3,  3,  3,  1,
         3,  7,  3,  3, 11,  7,  3,  3,  7,  3,  3, 19,  3,  3,  3,  3,
         3,  3, 11,  3,  3,  3,  3,  7,  3,  3,  3,  3,  3,  3,  3,  3,
...
         3,  3,  7,  3,  3,  7,  7,  3,  3,  3,  7,  3,  7,  3,  7,  3,
         3,  3,  3,  3,  3,  7,  7,  3,  3, 19,  3,  3,  3,  3,  7,  3,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  7,  3,  3,  3,
         3,  3,  3,  3, 11,  3,  3,  3, 11,  3],
       [ 3,  3,  3,  3, 11,  7,  3,  3,  3, 15,  3,  3,  3, 11,  3,  3,
         3,  7,  3,  3,  3,  3,  3,  3,  3, 11,  7,  3,  3,  3,  3,  3,
         3,  3,  3,  3,  3,  7,  3,  3,  3,  3,  7,  3,  1,  3,  3,  3,
        11,  3,  3,  3,  3,  3,  3,  3,  3,  1,  3,  3,  3,  3,  3,  7,
         7,  3,  7,  7,  3,  3,  3, 11,  3,  3,  3,  3,  3,  3,  7,  7,
         3,  7,  3,  3,  3,  3,  7,  3,  7,  3,  7,  3,  3,  3, 11,  3,
         3,  3, 15,  3,  3,  7,  3,  3,  3,  3,  3,  3,  1,  7,  3,  3,
         3,  3,  1,  3,  3,  3,  7,  3,  3,  3,  3,  3,  7,  7,  3, 15,
         3,  3,  3,  7,  3,  3,  3,  3,  3,  3,  3, 11,  3,  3,  7,  3,
         3,  7,  3,  3,  1,  3,  3,  3,  3,  7,  7,  3,  1,  3,  1,  3,
         3,  3,  3,  7,  3,  3,  3,  3,  3,  3,  3,  7,  7,  3,  3,  3,
         3,  3,  3,  3,  3,  3, 11,  3,  3,  3,  7,  7,  3,  3,  3,  3,
         3,  3, 11,  3,  3,  3,  3,  3,  3, 11,  3,  1,  3,  7,  3,  3,
         3,  7,  3,  3,  3,  7,  3,  3,  3,  1,  3,  3,  3,  3,  3,  3,
         7,  3,  7,  3,  7,  3,  3,  3,  3,  3,  3,  3,  1, 11,  3,  3,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8923 0.8923 ... 0.9726 0.9726</div><input id='attrs-d89b3b77-3e80-45b6-b827-2907ba744247' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d89b3b77-3e80-45b6-b827-2907ba744247' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5aedd87d-c95c-4f74-b69c-ce9051979512' class='xr-var-data-in' type='checkbox'><label for='data-5aedd87d-c95c-4f74-b69c-ce9051979512' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
        0.89234976, 0.89234976, 0.89234976, 0.89234976, 0.89234976,
...
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ,
        0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 , 0.9726011 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tree_depth</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>2 3 3 2 2 2 3 2 ... 2 2 2 2 2 2 2 2</div><input id='attrs-259e893a-822d-4bd0-8bdd-7ca498b1ebc5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-259e893a-822d-4bd0-8bdd-7ca498b1ebc5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c6b81d62-71d7-4ed7-9c62-621cbe23be6a' class='xr-var-data-in' type='checkbox'><label for='data-c6b81d62-71d7-4ed7-9c62-621cbe23be6a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2, 3, 3, 2, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3,
        2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 3, 2, 2, 3,
        3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3,
        3, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 1, 3, 3, 2, 2, 3, 2,
        2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 2,
        3, 2, 1, 3, 3, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3,
        3, 2, 2, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2,
        3, 2, 3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 1, 2, 3, 3, 2, 2, 2, 2,
        3, 2, 1, 2, 1, 2, 2, 3, 2, 2, 4, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2,
        2, 4, 2, 3, 1, 2, 1, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3,
        3, 3, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2,
        2, 3, 3, 2, 2, 2, 3, 2],
       [2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 4,
        2, 2, 2, 2, 2, 3, 2, 2, 2, 1, 2, 3, 2, 2, 4, 3, 2, 2, 3, 2, 2, 5,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3,
        2, 3, 2, 1, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3,
        2, 3, 4, 2, 2, 1, 2, 4, 2, 2, 2, 2, 2, 2, 3, 2, 4, 2, 2, 3, 2, 2,
        3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 3,
        2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3,
...
        2, 2, 4, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 2, 2, 3,
        2, 3, 2, 3, 2, 1, 3, 2, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2, 2, 3, 2, 2,
        2, 3, 2, 3, 2, 2, 2, 4, 2, 1, 2, 2, 2, 3, 4, 2, 2, 4, 2, 2, 2, 2,
        2, 2, 1, 3, 1, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 1, 2, 2, 3, 2, 2, 3,
        3, 2, 2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 5, 2, 2,
        2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2,
        2, 2, 4, 2, 2, 2, 4, 2],
       [2, 2, 2, 2, 4, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 2,
        2, 2, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2,
        1, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3, 2,
        3, 3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2,
        3, 2, 3, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 3, 2, 2, 2, 2, 2, 2, 1, 3,
        2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 2, 4, 2, 2, 2, 3,
        2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 3, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 3,
        3, 2, 1, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2,
        2, 2, 2, 4, 2, 1, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 1, 2, 2,
        2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 4, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f243f82d-135e-4dd2-a336-bcaec5842206' class='xr-section-summary-in' type='checkbox'  ><label for='section-f243f82d-135e-4dd2-a336-bcaec5842206' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-f93067bd-1921-48f1-8be6-1dd4682a539c' class='xr-index-data-in' type='checkbox'/><label for='index-f93067bd-1921-48f1-8be6-1dd4682a539c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-b1adfc64-8964-4f8b-9d99-3398f3007e2c' class='xr-index-data-in' type='checkbox'/><label for='index-b1adfc64-8964-4f8b-9d99-3398f3007e2c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       240, 241, 242, 243, 244, 245, 246, 247, 248, 249],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=250))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6e7b755f-2edb-4797-a338-084c3c0b95bc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6e7b755f-2edb-4797-a338-084c3c0b95bc' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:36:20.441267+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-sections.group-sections {
  grid-template-columns: auto;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
.xr-wrap{width:700px!important;} </style>



### Tensorflow probability


```python
tfp_nuts_idata = model.fit(inference_method="tfp_nuts")
tfp_nuts_idata
```


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
</pre>







            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">

            <li class = "xr-section-item">
                  <input id="idata_posterior2521d38c-5bcc-4323-966e-b10358d587fd" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posterior2521d38c-5bcc-4323-966e-b10358d587fd" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 200kB
Dimensions:    (chain: 8, draw: 1000)
Coordinates:
  * chain      (chain) int64 64B 0 1 2 3 4 5 6 7
  * draw       (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
Data variables:
    Intercept  (chain, draw) float64 64kB -0.1597 0.2011 ... 0.1525 -0.171
    x          (chain, draw) float64 64kB 0.2515 0.4686 0.4884 ... 0.5085 0.4896
    y_sigma    (chain, draw) float64 64kB 0.9735 0.8969 0.8002 ... 0.9422 1.045
Attributes:
    created_at:                  2024-04-13T05:36:30.303342+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-03a65e07-11b5-4140-ae68-70faeec80f5e' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-03a65e07-11b5-4140-ae68-70faeec80f5e' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 8</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-a05ab42e-f7e0-4ea6-807f-bc3ccb5bc9e2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a05ab42e-f7e0-4ea6-807f-bc3ccb5bc9e2' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-42674d56-c7ae-4632-a7a8-1904c6b468a4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-42674d56-c7ae-4632-a7a8-1904c6b468a4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca559c70-7d39-4775-8229-534008e219d9' class='xr-var-data-in' type='checkbox'><label for='data-ca559c70-7d39-4775-8229-534008e219d9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-62d24d71-5a14-4322-82e6-d01fc390d52e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-62d24d71-5a14-4322-82e6-d01fc390d52e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-de404f1f-f776-4658-802f-b0d22d972d5c' class='xr-var-data-in' type='checkbox'><label for='data-de404f1f-f776-4658-802f-b0d22d972d5c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8eba8b78-c6fe-414f-a019-9cfe9f9c4b61' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8eba8b78-c6fe-414f-a019-9cfe9f9c4b61' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>Intercept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.1597 0.2011 ... 0.1525 -0.171</div><input id='attrs-d2ad55f8-4da8-4071-bbe1-1b9a83e49e61' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d2ad55f8-4da8-4071-bbe1-1b9a83e49e61' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dc8b2264-0aca-47f9-a5b6-4fda89dc6c3b' class='xr-var-data-in' type='checkbox'><label for='data-dc8b2264-0aca-47f9-a5b6-4fda89dc6c3b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.15967084,  0.20113676,  0.01971475, ...,  0.00794969,
         0.0279562 ,  0.03472592],
       [-0.06799971,  0.14593555, -0.00604452, ...,  0.00924657,
        -0.10235794, -0.10236953],
       [ 0.043034  ,  0.08600223,  0.16562574, ...,  0.05851938,
         0.00720315,  0.08258778],
       ...,
       [ 0.04807806,  0.11227424, -0.3172604 , ...,  0.02980962,
        -0.13681545,  0.19177451],
       [ 0.04374417, -0.0054294 ,  0.09305579, ...,  0.0232273 ,
        -0.04073809,  0.025925  ],
       [-0.07370367, -0.00152223,  0.06769584, ..., -0.09818811,
         0.15246738, -0.17104419]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.2515 0.4686 ... 0.5085 0.4896</div><input id='attrs-c367be69-bda7-4588-ab52-ff6d2e174daa' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c367be69-bda7-4588-ab52-ff6d2e174daa' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ce5f0659-6641-472a-9e93-1dfdb144b659' class='xr-var-data-in' type='checkbox'><label for='data-ce5f0659-6641-472a-9e93-1dfdb144b659' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.25153111, 0.4685625 , 0.48837809, ..., 0.28573626, 0.407775  ,
        0.38347135],
       [0.28165967, 0.36310827, 0.41225084, ..., 0.24255857, 0.45039439,
        0.4954714 ],
       [0.5386156 , 0.6228231 , 0.25313292, ..., 0.44280376, 0.4488854 ,
        0.25456354],
       ...,
       [0.45168195, 0.46344655, 0.17750331, ..., 0.30371223, 0.29536054,
        0.40431303],
       [0.41455145, 0.43166272, 0.35213661, ..., 0.36384472, 0.3917272 ,
        0.34092006],
       [0.20620881, 0.51263399, 0.44056489, ..., 0.25237815, 0.50845624,
        0.48960883]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y_sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9735 0.8969 ... 0.9422 1.045</div><input id='attrs-83b2818d-f8ce-4f2c-8c72-3be45a84cf94' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-83b2818d-f8ce-4f2c-8c72-3be45a84cf94' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2167c6f2-b5ae-4986-9f85-0e07cb21f27c' class='xr-var-data-in' type='checkbox'><label for='data-2167c6f2-b5ae-4986-9f85-0e07cb21f27c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.97352428, 0.89691108, 0.80020873, ..., 1.03087931, 0.84944049,
        0.84158909],
       [0.96226504, 0.92778234, 0.77909925, ..., 0.91397532, 1.00185137,
        0.9513834 ],
       [1.0042728 , 0.97580931, 0.94890477, ..., 0.92691038, 0.885916  ,
        1.01934012],
       ...,
       [0.88671137, 0.91944589, 1.00541185, ..., 0.96151472, 0.93478611,
        0.94631027],
       [0.8367989 , 0.84727656, 1.05992876, ..., 0.91519111, 0.90516942,
        0.9358838 ],
       [0.94127918, 0.89667586, 0.91173519, ..., 0.96184559, 0.94224608,
        1.04527058]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2d4a04ef-145a-4abd-b9ca-6e8f162e16fb' class='xr-section-summary-in' type='checkbox'  ><label for='section-2d4a04ef-145a-4abd-b9ca-6e8f162e16fb' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-d8177818-fbf2-4aef-b0b0-c1f328ad99b5' class='xr-index-data-in' type='checkbox'/><label for='index-d8177818-fbf2-4aef-b0b0-c1f328ad99b5' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-5f396ff4-56b5-4c2e-9b0e-3d4f283ec5b2' class='xr-index-data-in' type='checkbox'/><label for='index-5f396ff4-56b5-4c2e-9b0e-3d4f283ec5b2' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-94b8e3eb-de9c-4ae1-a189-f2570f6822e7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-94b8e3eb-de9c-4ae1-a189-f2570f6822e7' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:36:30.303342+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

            <li class = "xr-section-item">
                  <input id="idata_sample_statsbb0ea48d-3afb-4938-a7a3-c3b178e3f52f" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_statsbb0ea48d-3afb-4938-a7a3-c3b178e3f52f" class = "xr-section-summary">sample_stats</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 312kB
Dimensions:          (chain: 8, draw: 1000)
Coordinates:
  * chain            (chain) int64 64B 0 1 2 3 4 5 6 7
  * draw             (draw) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
Data variables:
    accept_ratio     (chain, draw) float64 64kB 0.8971 0.9944 ... 0.9174 0.8133
    diverging        (chain, draw) bool 8kB False False False ... False False
    is_accepted      (chain, draw) bool 8kB True True True ... True True True
    n_steps          (chain, draw) int32 32kB 7 7 7 1 7 7 7 3 ... 3 7 7 7 3 7 7
    step_size        (chain, draw) float64 64kB 0.534 0.534 0.534 ... nan nan
    target_log_prob  (chain, draw) float64 64kB -141.5 -141.7 ... -141.0 -143.2
    tune             (chain, draw) float64 64kB 0.0 0.0 0.0 0.0 ... nan nan nan
Attributes:
    created_at:                  2024-04-13T05:36:30.304788+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-1822ff78-ae36-44f3-a44d-db61ca23d8cf' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-1822ff78-ae36-44f3-a44d-db61ca23d8cf' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 8</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-ce93cfa9-3d5d-4db5-bb1e-3f23fc703e5d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ce93cfa9-3d5d-4db5-bb1e-3f23fc703e5d' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-5feda318-82d3-4908-a332-9aecd9879106' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5feda318-82d3-4908-a332-9aecd9879106' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f37a1783-4af7-47b2-92c3-9d03beee87bb' class='xr-var-data-in' type='checkbox'><label for='data-f37a1783-4af7-47b2-92c3-9d03beee87bb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-e52f298a-806b-4962-946c-4624174c63b9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e52f298a-806b-4962-946c-4624174c63b9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3ed52901-5487-4f78-992b-3ef74bb4436b' class='xr-var-data-in' type='checkbox'><label for='data-3ed52901-5487-4f78-992b-3ef74bb4436b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-abde3658-b2e7-4970-8a4a-d5767562914d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-abde3658-b2e7-4970-8a4a-d5767562914d' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>accept_ratio</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8971 0.9944 ... 0.9174 0.8133</div><input id='attrs-6813625d-d516-41af-9b78-01880df88be7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6813625d-d516-41af-9b78-01880df88be7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-687a9bc5-98fe-485e-9753-1315f2fc35d0' class='xr-var-data-in' type='checkbox'><label for='data-687a9bc5-98fe-485e-9753-1315f2fc35d0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.89706765, 0.99441475, 0.80397664, ..., 0.99407977, 1.        ,
        0.73958291],
       [0.99821982, 0.95159754, 0.77731848, ..., 0.98139297, 0.91789348,
        0.96456953],
       [0.76824526, 0.92239538, 1.        , ..., 0.94414437, 0.91605876,
        0.92334246],
       ...,
       [0.99710475, 0.99154725, 0.58953539, ..., 1.        , 0.92397302,
        0.99338491],
       [0.98669117, 0.98477039, 0.95831938, ..., 0.92092812, 0.96842841,
        0.95013437],
       [0.91842649, 0.75186373, 0.99689159, ..., 1.        , 0.9173519 ,
        0.81331846]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-5d7428ae-a89b-4e3b-afbb-b97aacb6f5ca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5d7428ae-a89b-4e3b-afbb-b97aacb6f5ca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6d6b2688-5cf1-40b5-b17c-4d3f7b8ef951' class='xr-var-data-in' type='checkbox'><label for='data-6d6b2688-5cf1-40b5-b17c-4d3f7b8ef951' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>is_accepted</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>True True True ... True True True</div><input id='attrs-d5141200-d3fd-4d38-bb8c-7f7e91c23eee' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d5141200-d3fd-4d38-bb8c-7f7e91c23eee' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-80fd3099-614b-4d15-ad93-977254f5ff51' class='xr-var-data-in' type='checkbox'><label for='data-80fd3099-614b-4d15-ad93-977254f5ff51' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       ...,
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True],
       [ True,  True,  True, ...,  True,  True,  True]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>n_steps</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>7 7 7 1 7 7 7 3 ... 3 3 7 7 7 3 7 7</div><input id='attrs-cf8899a1-53c6-4d55-8a39-9c3d8ddab92f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cf8899a1-53c6-4d55-8a39-9c3d8ddab92f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-365dda8c-12bd-4472-a7aa-f1a8b53d181e' class='xr-var-data-in' type='checkbox'><label for='data-365dda8c-12bd-4472-a7aa-f1a8b53d181e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 7,  7,  7, ...,  7,  7,  7],
       [ 7,  7,  3, ...,  7,  7,  3],
       [ 3,  3,  7, ...,  3, 15,  7],
       ...,
       [ 7,  7,  3, ...,  3,  7,  7],
       [ 7,  7,  7, ...,  7,  3,  3],
       [ 7,  7,  3, ...,  3,  7,  7]], dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.534 0.534 0.534 ... nan nan nan</div><input id='attrs-f48a4f7f-0059-4e3a-aa9b-bf725815dbe2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f48a4f7f-0059-4e3a-aa9b-bf725815dbe2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-698e3d6d-7c6f-4d67-b564-5655b16ead8b' class='xr-var-data-in' type='checkbox'><label for='data-698e3d6d-7c6f-4d67-b564-5655b16ead8b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.53403598, 0.53403598, 0.53403598, ..., 0.53403598, 0.53403598,
        0.53403598],
       [       nan,        nan,        nan, ...,        nan,        nan,
               nan],
       [       nan,        nan,        nan, ...,        nan,        nan,
               nan],
       ...,
       [       nan,        nan,        nan, ...,        nan,        nan,
               nan],
       [       nan,        nan,        nan, ...,        nan,        nan,
               nan],
       [       nan,        nan,        nan, ...,        nan,        nan,
               nan]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>target_log_prob</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-141.5 -141.7 ... -141.0 -143.2</div><input id='attrs-88edd36d-eda2-44f5-87ba-ed25a36b477e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-88edd36d-eda2-44f5-87ba-ed25a36b477e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6f97cbb1-e055-44a1-8438-7bac5438b296' class='xr-var-data-in' type='checkbox'><label for='data-6f97cbb1-e055-44a1-8438-7bac5438b296' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-141.50324612, -141.68928918, -142.88857248, ..., -140.47032751,
        -140.28212037, -140.3879345 ],
       [-140.01375434, -140.13505676, -143.13641583, ..., -139.99147863,
        -141.05078466, -141.26466845],
       [-141.12400308, -142.52999266, -141.18044662, ..., -139.62740626,
        -139.97278032, -140.75200575],
       ...,
       [-139.95465107, -140.14159326, -146.22556521, ..., -139.52421787,
        -140.79404952, -140.86677285],
       [-140.66929954, -140.59550579, -141.05595747, ..., -139.29239833,
        -139.67236895, -139.28664301],
       [-140.73991807, -140.71540356, -139.69662159, ..., -140.53100323,
        -140.99477405, -143.18216599]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tune</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.0 0.0 0.0 ... nan nan nan nan</div><input id='attrs-93b6c868-756d-481a-a75a-43e7d3c9ff12' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-93b6c868-756d-481a-a75a-43e7d3c9ff12' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eacc0d4f-4289-42bd-979a-803205c56481' class='xr-var-data-in' type='checkbox'><label for='data-eacc0d4f-4289-42bd-979a-803205c56481' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [nan, nan, nan, ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan],
       ...,
       [nan, nan, nan, ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan],
       [nan, nan, nan, ..., nan, nan, nan]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-080e973e-b967-4343-9b08-80c694199431' class='xr-section-summary-in' type='checkbox'  ><label for='section-080e973e-b967-4343-9b08-80c694199431' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-7267512d-59ea-4da9-bc34-124c68dfd607' class='xr-index-data-in' type='checkbox'/><label for='index-7267512d-59ea-4da9-bc34-124c68dfd607' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-974ae6a2-565c-4518-97a8-a95afc91dae3' class='xr-index-data-in' type='checkbox'/><label for='index-974ae6a2-565c-4518-97a8-a95afc91dae3' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bbd93bc7-10b7-4b86-ad0d-bae11f2296c9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bbd93bc7-10b7-4b86-ad0d-bae11f2296c9' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:36:30.304788+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-sections.group-sections {
  grid-template-columns: auto;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
.xr-wrap{width:700px!important;} </style>



### NumPyro


```python
numpyro_nuts_idata = model.fit(inference_method="numpyro_nuts")
numpyro_nuts_idata
```

    sample: 100%|██████████| 1500/1500 [00:02<00:00, 599.25it/s]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
</pre>







            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">

            <li class = "xr-section-item">
                  <input id="idata_posterior63dfe943-d83d-4eb1-beca-ee6691fcaa62" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posterior63dfe943-d83d-4eb1-beca-ee6691fcaa62" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 200kB
Dimensions:    (chain: 8, draw: 1000)
Coordinates:
  * chain      (chain) int64 64B 0 1 2 3 4 5 6 7
  * draw       (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
Data variables:
    Intercept  (chain, draw) float64 64kB 0.0004764 0.02933 ... 0.1217 0.1668
    x          (chain, draw) float64 64kB 0.3836 0.6556 0.2326 ... 0.48 0.5808
    y_sigma    (chain, draw) float64 64kB 0.8821 0.9604 0.9652 ... 0.9063 0.9184
Attributes:
    created_at:                  2024-04-13T05:36:33.599519+00:00
    arviz_version:               0.18.0
    inference_library:           numpyro
    inference_library_version:   0.14.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-cba7f07b-fbe8-4907-8a26-8ddc5a3ba031' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cba7f07b-fbe8-4907-8a26-8ddc5a3ba031' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 8</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-bbc9ef63-a437-4089-858f-d31b3dac6d76' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bbc9ef63-a437-4089-858f-d31b3dac6d76' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-b7138edb-f9b9-4d2f-aba8-5ba5a488c310' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b7138edb-f9b9-4d2f-aba8-5ba5a488c310' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e139ccd4-b306-496e-8659-57d5df790e2a' class='xr-var-data-in' type='checkbox'><label for='data-e139ccd4-b306-496e-8659-57d5df790e2a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-1f1ed251-7ed7-416f-a7d4-a2f0f7d30d66' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1f1ed251-7ed7-416f-a7d4-a2f0f7d30d66' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-449c7510-85bb-4791-999c-9b5ad8bd9148' class='xr-var-data-in' type='checkbox'><label for='data-449c7510-85bb-4791-999c-9b5ad8bd9148' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ce926eaf-e73b-4e45-93a2-1aa360f9b021' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ce926eaf-e73b-4e45-93a2-1aa360f9b021' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>Intercept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0004764 0.02933 ... 0.1217 0.1668</div><input id='attrs-f9aecc69-2b74-4830-b807-0d174da32683' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f9aecc69-2b74-4830-b807-0d174da32683' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-48878a7e-5b72-49a0-a8ec-c90727cb6740' class='xr-var-data-in' type='checkbox'><label for='data-48878a7e-5b72-49a0-a8ec-c90727cb6740' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.00047641,  0.02933426, -0.19069113, ..., -0.04607573,
         0.14408182, -0.07016404],
       [-0.06542953,  0.09001091,  0.03811068, ...,  0.03170986,
         0.20861147,  0.18706729],
       [-0.08028089,  0.03625393,  0.05650287, ..., -0.0093195 ,
         0.01912548,  0.00214345],
       ...,
       [ 0.18184083,  0.07906243,  0.06388914, ..., -0.07055763,
         0.10986417,  0.09622923],
       [-0.02521011,  0.15830259, -0.10214413, ...,  0.01471807,
         0.10706226,  0.07562878],
       [-0.02468806, -0.03414193, -0.06678234, ...,  0.08710519,
         0.12166933,  0.16679929]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3836 0.6556 ... 0.48 0.5808</div><input id='attrs-b46d8eda-88ef-4dc3-a542-23a586056b9a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b46d8eda-88ef-4dc3-a542-23a586056b9a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d3de08ba-df44-44ea-9e22-3521fd18a0e6' class='xr-var-data-in' type='checkbox'><label for='data-d3de08ba-df44-44ea-9e22-3521fd18a0e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.38361084, 0.65556045, 0.23260059, ..., 0.65580692, 0.44095681,
        0.22838517],
       [0.30187358, 0.46285734, 0.31814527, ..., 0.38133365, 0.32358724,
        0.37070791],
       [0.44410357, 0.42831529, 0.3990648 , ..., 0.37993575, 0.40377358,
        0.42804019],
       ...,
       [0.49080324, 0.20770949, 0.12142607, ..., 0.44054445, 0.38924394,
        0.38167612],
       [0.34590162, 0.30144285, 0.45780034, ..., 0.44424986, 0.52104263,
        0.45543543],
       [0.23738988, 0.68021684, 0.05589656, ..., 0.42147165, 0.48000601,
        0.58081686]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y_sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8821 0.9604 ... 0.9063 0.9184</div><input id='attrs-4b8b7002-0ac5-4384-9c8c-c6b48cef4f2e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4b8b7002-0ac5-4384-9c8c-c6b48cef4f2e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2f30b7b2-694c-4372-b7a7-9c96d40f6f1c' class='xr-var-data-in' type='checkbox'><label for='data-2f30b7b2-694c-4372-b7a7-9c96d40f6f1c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.88211276, 0.96036122, 0.96524442, ..., 0.94362502, 1.00228679,
        0.88249142],
       [0.93345676, 0.85184129, 1.07135935, ..., 0.92649839, 0.86831784,
        0.92890112],
       [0.973364  , 1.04138907, 0.96240687, ..., 0.9564475 , 1.0092212 ,
        0.87607713],
       ...,
       [1.03355029, 0.98103228, 0.92902834, ..., 0.83197448, 0.99111854,
        0.92967952],
       [0.88101923, 1.0226885 , 0.87217557, ..., 0.94028186, 0.88687764,
        0.85291778],
       [0.98596365, 0.91083125, 0.9972831 , ..., 0.86419289, 0.90625839,
        0.91841349]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-abf13c70-bf1e-479d-9f41-e2cf3f5a84f8' class='xr-section-summary-in' type='checkbox'  ><label for='section-abf13c70-bf1e-479d-9f41-e2cf3f5a84f8' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-d4402271-acee-45d6-90cf-1b6538866cc3' class='xr-index-data-in' type='checkbox'/><label for='index-d4402271-acee-45d6-90cf-1b6538866cc3' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-49d86a28-2006-41d2-be0c-31ee89128095' class='xr-index-data-in' type='checkbox'/><label for='index-49d86a28-2006-41d2-be0c-31ee89128095' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-42337414-9045-4f5b-8dc2-d3bfd798a36c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-42337414-9045-4f5b-8dc2-d3bfd798a36c' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:36:33.599519+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.14.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

            <li class = "xr-section-item">
                  <input id="idata_sample_stats104700a1-9b3f-44a5-ac58-210c643f8c24" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_stats104700a1-9b3f-44a5-ac58-210c643f8c24" class = "xr-section-summary">sample_stats</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 400kB
Dimensions:          (chain: 8, draw: 1000)
Coordinates:
  * chain            (chain) int64 64B 0 1 2 3 4 5 6 7
  * draw             (draw) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
Data variables:
    acceptance_rate  (chain, draw) float64 64kB 0.9366 0.3542 ... 0.9903 0.8838
    diverging        (chain, draw) bool 8kB False False False ... False False
    energy           (chain, draw) float64 64kB 140.3 145.4 ... 141.0 142.6
    lp               (chain, draw) float64 64kB 139.6 143.3 ... 140.5 142.5
    n_steps          (chain, draw) int64 64kB 3 3 7 3 7 1 3 3 ... 11 7 3 3 3 3 3
    step_size        (chain, draw) float64 64kB 0.8891 0.8891 ... 0.7595 0.7595
    tree_depth       (chain, draw) int64 64kB 2 2 3 2 3 1 2 2 ... 4 3 2 2 2 2 2
Attributes:
    created_at:                  2024-04-13T05:36:33.623197+00:00
    arviz_version:               0.18.0
    inference_library:           numpyro
    inference_library_version:   0.14.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-6c5a6f83-2206-4d42-a6b2-82b610436a2f' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6c5a6f83-2206-4d42-a6b2-82b610436a2f' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 8</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-4eec70aa-075c-4ab3-a752-6f7c4aa910db' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4eec70aa-075c-4ab3-a752-6f7c4aa910db' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7</div><input id='attrs-dabab906-880f-4a49-a496-e8e828a942e5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dabab906-880f-4a49-a496-e8e828a942e5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78266d51-966c-496d-9f4c-8d11f5bb63c2' class='xr-var-data-in' type='checkbox'><label for='data-78266d51-966c-496d-9f4c-8d11f5bb63c2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-c3d196aa-d34f-409e-bb7f-91e5a64b293d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c3d196aa-d34f-409e-bb7f-91e5a64b293d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a37d2667-62c3-4bae-b3a4-0b25d4ca2fa1' class='xr-var-data-in' type='checkbox'><label for='data-a37d2667-62c3-4bae-b3a4-0b25d4ca2fa1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1f572ae9-84ba-480a-9c9f-0ca7f0422bfe' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1f572ae9-84ba-480a-9c9f-0ca7f0422bfe' class='xr-section-summary' >Data variables: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>acceptance_rate</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9366 0.3542 ... 0.9903 0.8838</div><input id='attrs-63fb632f-4c36-4271-a2fb-37c097bfaa71' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-63fb632f-4c36-4271-a2fb-37c097bfaa71' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5330bb45-b0c4-44ec-82b7-81e50781fe23' class='xr-var-data-in' type='checkbox'><label for='data-5330bb45-b0c4-44ec-82b7-81e50781fe23' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.93661577, 0.35419612, 0.99435023, ..., 0.59003267, 1.        ,
        0.96452433],
       [0.9974338 , 0.86250112, 0.95945138, ..., 0.78208773, 0.79906599,
        1.        ],
       [0.96468642, 0.97525962, 0.98495362, ..., 0.9775774 , 0.86156602,
        0.89713276],
       ...,
       [0.96610793, 0.98086156, 0.89084022, ..., 0.95772457, 0.70497474,
        0.99836264],
       [0.99806445, 0.76652499, 0.98528715, ..., 0.87560309, 0.81183609,
        1.        ],
       [0.99527811, 0.68120359, 1.        , ..., 1.        , 0.99026377,
        0.88375821]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-5cc54c06-6d4e-4d49-b1a9-cc17b22c7965' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5cc54c06-6d4e-4d49-b1a9-cc17b22c7965' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a99570c7-74cc-4230-ab9e-d2ce00e13e58' class='xr-var-data-in' type='checkbox'><label for='data-a99570c7-74cc-4230-ab9e-d2ce00e13e58' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>140.3 145.4 144.9 ... 141.0 142.6</div><input id='attrs-2c475ce0-3dec-4153-837c-41ca733b5878' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2c475ce0-3dec-4153-837c-41ca733b5878' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-38733b29-f36f-4fa5-a3b3-375e1cebc12f' class='xr-var-data-in' type='checkbox'><label for='data-38733b29-f36f-4fa5-a3b3-375e1cebc12f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[140.3467186 , 145.35611236, 144.90015851, ..., 144.55362515,
        144.91727277, 142.02764348],
       [141.22868211, 142.17975426, 142.8293087 , ..., 142.28333599,
        142.31235728, 142.52446205],
       [144.0720805 , 141.47372901, 140.93450225, ..., 140.18268471,
        140.91505172, 141.58829808],
       ...,
       [142.62435102, 143.2286207 , 142.80972656, ..., 144.52562661,
        145.55539646, 140.15311622],
       [142.10722077, 142.2552196 , 142.37900379, ..., 140.54316684,
        141.75774836, 141.58904973],
       [140.82607551, 145.03403209, 145.2241475 , ..., 141.6051566 ,
        140.9854462 , 142.60201809]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lp</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>139.6 143.3 142.2 ... 140.5 142.5</div><input id='attrs-1c165582-9dca-43d4-b5c5-dbf497a8b986' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c165582-9dca-43d4-b5c5-dbf497a8b986' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-067f0fe0-ba21-4e21-be0e-4faf15f6654a' class='xr-var-data-in' type='checkbox'><label for='data-067f0fe0-ba21-4e21-be0e-4faf15f6654a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[139.63264222, 143.32720395, 142.23861021, ..., 143.97346231,
        140.67090827, 140.87592102],
       [139.81127204, 140.89739965, 141.16918465, ..., 139.28792859,
        142.23961923, 140.80131401],
       [140.45054751, 140.62345763, 139.48418855, ..., 139.42120354,
        139.97119948, 139.9471655 ],
       ...,
       [141.9099355 , 140.83346483, 142.30482492, ..., 141.78502364,
        140.03250066, 139.57119158],
       [139.72932132, 141.21433158, 141.46193109, ..., 139.62199619,
        141.12185494, 140.69980318],
       [140.29191412, 144.98589172, 143.74390375, ..., 140.2609778 ,
        140.48172152, 142.47099219]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>n_steps</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>3 3 7 3 7 1 3 3 ... 11 7 3 3 3 3 3</div><input id='attrs-8f0fb13d-0ca1-4443-b80e-2fb9960d8861' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8f0fb13d-0ca1-4443-b80e-2fb9960d8861' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c2161ef8-f671-404d-a04f-fddd1ea03c3f' class='xr-var-data-in' type='checkbox'><label for='data-c2161ef8-f671-404d-a04f-fddd1ea03c3f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[3, 3, 7, ..., 3, 3, 7],
       [7, 3, 7, ..., 3, 3, 7],
       [3, 7, 3, ..., 3, 3, 3],
       ...,
       [7, 7, 7, ..., 3, 7, 7],
       [3, 7, 7, ..., 3, 3, 7],
       [7, 7, 3, ..., 3, 3, 3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8891 0.8891 ... 0.7595 0.7595</div><input id='attrs-44f46511-a261-4c41-ad00-41cfdadd2963' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-44f46511-a261-4c41-ad00-41cfdadd2963' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f2c4e1fc-6ceb-49d5-b40b-625fe7ad5972' class='xr-var-data-in' type='checkbox'><label for='data-f2c4e1fc-6ceb-49d5-b40b-625fe7ad5972' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.8891145 , 0.8891145 , 0.8891145 , ..., 0.8891145 , 0.8891145 ,
        0.8891145 ],
       [0.70896111, 0.70896111, 0.70896111, ..., 0.70896111, 0.70896111,
        0.70896111],
       [0.8087902 , 0.8087902 , 0.8087902 , ..., 0.8087902 , 0.8087902 ,
        0.8087902 ],
       ...,
       [0.69745418, 0.69745418, 0.69745418, ..., 0.69745418, 0.69745418,
        0.69745418],
       [0.88034552, 0.88034552, 0.88034552, ..., 0.88034552, 0.88034552,
        0.88034552],
       [0.7595237 , 0.7595237 , 0.7595237 , ..., 0.7595237 , 0.7595237 ,
        0.7595237 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tree_depth</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>2 2 3 2 3 1 2 2 ... 2 4 3 2 2 2 2 2</div><input id='attrs-a27e40e8-99cc-4e2e-b8f8-5fd17bc0162e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a27e40e8-99cc-4e2e-b8f8-5fd17bc0162e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fc009c8c-21d8-4827-bd17-65b01aaf1ca7' class='xr-var-data-in' type='checkbox'><label for='data-fc009c8c-21d8-4827-bd17-65b01aaf1ca7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2, 2, 3, ..., 2, 2, 3],
       [3, 2, 3, ..., 2, 2, 3],
       [2, 3, 2, ..., 2, 2, 2],
       ...,
       [3, 3, 3, ..., 2, 3, 3],
       [2, 3, 3, ..., 2, 2, 3],
       [3, 3, 2, ..., 2, 2, 2]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-28c56125-e9d9-445f-a7d7-71b93b4c82cc' class='xr-section-summary-in' type='checkbox'  ><label for='section-28c56125-e9d9-445f-a7d7-71b93b4c82cc' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-bc0fc5a8-497f-4c6d-b1c8-6bdeb1d7f203' class='xr-index-data-in' type='checkbox'/><label for='index-bc0fc5a8-497f-4c6d-b1c8-6bdeb1d7f203' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-b0f684ed-3795-4aa9-b555-cf97a184bcd6' class='xr-index-data-in' type='checkbox'/><label for='index-b0f684ed-3795-4aa9-b555-cf97a184bcd6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       990, 991, 992, 993, 994, 995, 996, 997, 998, 999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=1000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-fadd64a0-2e90-4de0-b0fd-9426d26939e9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-fadd64a0-2e90-4de0-b0fd-9426d26939e9' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:36:33.623197+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>inference_library :</span></dt><dd>numpyro</dd><dt><span>inference_library_version :</span></dt><dd>0.14.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-sections.group-sections {
  grid-template-columns: auto;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
.xr-wrap{width:700px!important;} </style>



### flowMC


```python
flowmc_idata = model.fit(inference_method="flowmc_realnvp_hmc")
flowmc_idata
```

    No autotune found, use input sampler_params
    Training normalizing flow


    Tuning global sampler: 100%|██████████| 5/5 [00:51<00:00, 10.37s/it]


    Starting Production run


    Production run: 100%|██████████| 5/5 [00:00<00:00, 14.38it/s]


    Output()


    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
</pre>







            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">

            <li class = "xr-section-item">
                  <input id="idata_posteriorfd493682-c762-4407-81b4-1465c355352f" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posteriorfd493682-c762-4407-81b4-1465c355352f" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 244kB
Dimensions:    (chain: 20, draw: 500)
Coordinates:
  * chain      (chain) int64 160B 0 1 2 3 4 5 6 7 8 ... 12 13 14 15 16 17 18 19
  * draw       (draw) int64 4kB 0 1 2 3 4 5 6 7 ... 493 494 495 496 497 498 499
Data variables:
    Intercept  (chain, draw) float64 80kB -0.07404 -0.07404 ... -0.1455 0.09545
    x          (chain, draw) float64 80kB 0.4401 0.4401 0.3533 ... 0.6115 0.3824
    y_sigma    (chain, draw) float64 80kB 0.9181 0.9181 0.9732 ... 1.049 0.9643
Attributes:
    created_at:                  2024-04-13T05:37:29.798250+00:00
    arviz_version:               0.18.0
    modeling_interface:          bambi
    modeling_interface_version:  0.13.1.dev25+g1e7f677e.d20240413</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-6b0e99d9-d9c0-4734-93cd-c3992bf43d98' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6b0e99d9-d9c0-4734-93cd-c3992bf43d98' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 20</li><li><span class='xr-has-index'>draw</span>: 500</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6e7a079f-8d65-45a1-a38e-18e156a98f03' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6e7a079f-8d65-45a1-a38e-18e156a98f03' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 14 15 16 17 18 19</div><input id='attrs-65d657b3-c026-4bd9-9589-12edf805b4a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-65d657b3-c026-4bd9-9589-12edf805b4a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d3f3618e-e47f-4473-9fa6-42b85ca9d0e8' class='xr-var-data-in' type='checkbox'><label for='data-d3f3618e-e47f-4473-9fa6-42b85ca9d0e8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 495 496 497 498 499</div><input id='attrs-b5f24446-f5a4-4a94-af9b-32308383a7e9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b5f24446-f5a4-4a94-af9b-32308383a7e9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c199d8bb-0094-425a-8d5d-6468dc766e9b' class='xr-var-data-in' type='checkbox'><label for='data-c199d8bb-0094-425a-8d5d-6468dc766e9b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 497, 498, 499])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-611d23f1-3e9d-4e5d-9ddb-0ae33c5f1a35' class='xr-section-summary-in' type='checkbox'  checked><label for='section-611d23f1-3e9d-4e5d-9ddb-0ae33c5f1a35' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>Intercept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.07404 -0.07404 ... 0.09545</div><input id='attrs-c09a3195-6767-4e21-9b5d-b744d165f4a5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c09a3195-6767-4e21-9b5d-b744d165f4a5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9f2a8850-797a-4330-91bb-2ba6323c33f7' class='xr-var-data-in' type='checkbox'><label for='data-9f2a8850-797a-4330-91bb-2ba6323c33f7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.07403956, -0.07403956, -0.08175103, ..., -0.07505013,
        -0.06280065, -0.06280065],
       [ 0.15335447,  0.15335447,  0.12713003, ..., -0.02037623,
         0.06826204,  0.01933312],
       [ 0.00658099,  0.00658099,  0.00658099, ..., -0.04271278,
        -0.04271278, -0.09780863],
       ...,
       [ 0.00629487,  0.01048304, -0.03193874, ...,  0.13237167,
         0.08595727,  0.01442809],
       [ 0.05972149,  0.02490161, -0.00084261, ...,  0.06751994,
        -0.15926318, -0.15926318],
       [ 0.23012418,  0.25630661,  0.23839857, ...,  0.07975465,
        -0.14554836,  0.09545347]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.4401 0.4401 ... 0.6115 0.3824</div><input id='attrs-396f8272-6f49-4c02-92af-453188a10cc7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-396f8272-6f49-4c02-92af-453188a10cc7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cff163e5-af6b-426c-892f-aab0d4cd8af5' class='xr-var-data-in' type='checkbox'><label for='data-cff163e5-af6b-426c-892f-aab0d4cd8af5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.44013865, 0.44013865, 0.35326474, ..., 0.30371128, 0.28793687,
        0.28793687],
       [0.45569737, 0.45569737, 0.55350522, ..., 0.37498493, 0.45850535,
        0.40671648],
       [0.24971734, 0.24971734, 0.24971734, ..., 0.19912158, 0.19912158,
        0.46992411],
       ...,
       [0.44071041, 0.47684243, 0.35786393, ..., 0.37932871, 0.31101246,
        0.25090813],
       [0.43305649, 0.19703032, 0.21622992, ..., 0.39021766, 0.35161734,
        0.35161734],
       [0.52832789, 0.50016524, 0.19504762, ..., 0.25411208, 0.61146903,
        0.38243421]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>y_sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9181 0.9181 ... 1.049 0.9643</div><input id='attrs-55633323-4c6e-4f97-8ed4-8a5f5ee206db' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-55633323-4c6e-4f97-8ed4-8a5f5ee206db' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78c7a69c-91ba-4f82-8c68-de84ca5f7e3f' class='xr-var-data-in' type='checkbox'><label for='data-78c7a69c-91ba-4f82-8c68-de84ca5f7e3f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.91812597, 0.91812597, 0.97317218, ..., 0.87193011, 0.98202548,
        0.98202548],
       [0.92619283, 0.92619283, 0.89113835, ..., 1.00239178, 0.93585383,
        0.93328517],
       [0.96032594, 0.96032594, 0.96032594, ..., 0.90617649, 0.90617649,
        0.95241728],
       ...,
       [1.01337917, 0.96203307, 0.81645174, ..., 1.00979845, 1.07249345,
        0.9165658 ],
       [0.97087149, 0.91876884, 0.87129204, ..., 1.09021385, 1.0093326 ,
        1.0093326 ],
       [0.89533883, 0.91515164, 1.07248889, ..., 0.95594426, 1.04908995,
        0.96426064]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-3a355d38-74f0-4f52-9dde-b367f5dbd846' class='xr-section-summary-in' type='checkbox'  ><label for='section-3a355d38-74f0-4f52-9dde-b367f5dbd846' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-6a646ce1-c92d-44e3-9a90-2307cc6aba06' class='xr-index-data-in' type='checkbox'/><label for='index-6a646ce1-c92d-44e3-9a90-2307cc6aba06' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-39bc03e7-ab5e-46a9-926a-89805301e4d0' class='xr-index-data-in' type='checkbox'/><label for='index-39bc03e7-ab5e-46a9-926a-89805301e4d0' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       490, 491, 492, 493, 494, 495, 496, 497, 498, 499],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=500))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7c932e09-4416-4d0a-8464-43b8e0921ad6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7c932e09-4416-4d0a-8464-43b8e0921ad6' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2024-04-13T05:37:29.798250+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.18.0</dd><dt><span>modeling_interface :</span></dt><dd>bambi</dd><dt><span>modeling_interface_version :</span></dt><dd>0.13.1.dev25+g1e7f677e.d20240413</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>

              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-sections.group-sections {
  grid-template-columns: auto;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
.xr-wrap{width:700px!important;} </style>



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
    
