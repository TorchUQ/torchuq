This repo is still under developement.  Please check back in a few weeks. 

## What is torchuq?

TorchUQ is your one-stop solution for uncertainty quantification (UQ). At its core, TorchUQ supports various representations of uncertainty (including probability, quantile, particle, ensemble, set, etc), and 

1. converts between different representations of uncertainty 

2. adjusts any uncertainty representation so it becomes calibrated/valid/multi-accuarate..... 

3. evaluates and visualizes the quality of uncertainty quantification 

TorchUQ is built on pytorch, and supports auto-differentiation for most of its functions. Current version only support a pytorch interface (i.e. all arrays have to be pytorch arrays). Support for numpy interface to come. 


## Why use torchuq? 

1. Torchuq is a **one-stop solution**. You can use one toolbox with a single consistent interface to evaluate predictions, or run popular calibration or conformal inference algorithms. Each function can be accomplished with 1-5 lines of code. 

2. Torchuq is based on pytorch, so all functions **support GPU** acceleration with no overhead (If a function receives GPU tensors as input, then it is automatically computed on GPU). Most functions are also **end-to-end differentiable** and can be incorporated into a deep learning pipeline. 

3. Torchuq includes a **large set of tutorials** to illustrate popular algorithms and evaluation metrics for uncertainty quantification. 

4. Torchuq comes with a large set of **benchmark datasets** used in recent UQ papers with a one-line interface to retrieve these datasets.

5. Torchuq is **well tested**. In the dev folder there are a large number of unit tests to ensure correctness. 

## How to start? 

This repo contains everything you need. The ``torchuq`` folder contains all the core functionality of torchuq, and the ``docs`` folder contains all the tutorials and documentation. 

A good way to start is to go through the tutorials in ``docs/tutorial``. All the tutorials are interactive jupyter notebooks. For non-interactive viewing on a browser see [link TBD]. 

To run the code, you can install the dependencies with the follwoing command 

`` pip install TBD ``
