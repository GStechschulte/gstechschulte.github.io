+++
title = 'Google Summer of Code - Final Report'
date = 2023-08-10T18:43:46+02:00
author = 'Gabriel Stechschulte'
categories = ['open-source']
draft = false
+++

![alt](bambi-logo.png)

My project "Better tools to interpret complex Bambi regression models" was completed under the organization of NumFOCUS, and mentors Tomás Capretto and Osvaldo Martin. Before I describe the project, objectives, and work completed, I would like to thank my mentors Tomás and Osvaldo for their precious time and support throughout the summer. They were always available and timely in communicating over Slack and GitHub, and provided valuable feedback during code reviews. Additionally, I would like to thank NumFOCUS and the Google Summer of Code (GSoC) program for providing the opportunity to work on such an open source project over the summer. It has been an invaluable experience, and I look forward to contributing to open source projects in the future.

# Project Description

Bayesian modeling has increased significantly in academia and industry over the past years thanks to the development of high quality and user friendly open source probabilistic programming languages (PPL) in Python and R. Of these is Bambi, a Python library built on top of the PyMC PPL, that makes it easy to specify complex generalized linear multilevel models using a formula notation similar to those found in R. However, as the model building portion of the Bayesian workflow becomes easier, the interpretation of these models has not.

To aid in model interpretability, Bambi (before this project) had a sub-package `plots` that supported _conditional adjusted predictions_ plots.
The original objective was to extend upon the existing plotting functionality of _conditional adjusted predictions_ by supporting the plotting of posterior predictive samples, and to provide the additional plotting functions _predictive comparisons_ and _predictive slopes_. However, after discussion with my mentors, and taking inspiration from [marginaleffects](https://marginaleffects.com), it was decided that in addition to the plotting functions, the `plots` sub-package should also have functions that allow the user to return the dataframe used for plotting. For example, calling `plot_slopes()` would plot the slopes, and calling `slopes()` would return the dataframe used to plot the slopes. To this extent, the `plots` sub-package was renamed to `interpret` to better reflect all of the supported functionality. These three features allow Bambi modelers to compute and interpret three different "quantities of interest" in a more automatic and effective manner.

Thus, the three main deliverables of this project were to add (or enhance) the following functions in the `interpret` sub-package. Additionally, for each feature, tests and documentation need to be added:

* Support posterior predictive samples (pps) in `plot_predictions()`
    * Write tests
    * Add documentation
* Add `comparisons()` and `plot_comparisons()`
    * Write tests
    * Add documentation
* Add `slopes()` and `plot_slopes()`
    * Write tests
    * Add documentation

# Work Completed

All three main deliverables (and their associated sub-deliverables) were completed (merged into the main branch of the Bambi repository) on time. In the table below, the name of the deliverable, link to the feature's pull request (PR), and link to the documentation are provided.

| Deliverable | Feature PR | Documentation PR |
|-------------|------------|------------------|
| Allow `plot_predictions` to plot posterior predictive samples | PR [#668](https://github.com/bambinos/bambi/pull/668) | PR [#670](https://bambinos.github.io/bambi/notebooks/plot_comparisons.html) |
| Add `comparisons` and `plot_comparisons` | PR [#684](https://github.com/bambinos/bambi/pull/684) | PR [#695](https://bambinos.github.io/bambi/notebooks/plot_comparisons.html) |
| Add `slopes` and `plot_slopes` | PR [#699](https://github.com/bambinos/bambi/pull/699) | PR [#701](https://bambinos.github.io/bambi/notebooks/plot_slopes.html) |

In order to quickly obtain a sense of the work completed over the GSoC program, it is probably best to view the documentation PRs. The documentation PRs consist of a Jupyter notebook that demonstrates: (1) why the new feature is useful in interpreting GLMs, (2) a brief overview of how the function computes the quantity of interest based on the user's inputs, and (3) a demonstration of the function using different GLMs and datasets.

# Future Work

As the PRs have been merged upstream, some Bambi users have already been utilizing the new sub-package. The feedback has been positive, and a GitHub [issue](https://github.com/bambinos/bambi/issues/703) has been opened to request additional functionality. Going forward, I will continue to contribute to Bambi, maintain the `interpret` sub-package, and interact with the Bambi community.
