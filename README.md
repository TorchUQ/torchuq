## Torchuq [logo here]

> TorchUQ is an extensive library for uncertainty quantification (UQ) based on pytorch.
> TorchUQ currently supports 10 representations for uncertainty, and around 50 different methods for uncertainty evaluation and visualization, calibration and conformal prediction. 

## Why TorchUQ 

Uncertainty quantification (UQ) --- prediction models should know what they do not know --- finds numerous applications in active learning, statistical inference, trustworthy machine learning, or in natural science and engineering applications that are rife with sources of uncertainty. TorchUQ aims to help both practitioners and researchers use UQ methods with ease. 

###  For practitioners

Torchuq provides an easy-to-use arsenal of uncertainty quantification methods with the following key features: 

**Plug and Play**: Simple unified interface to access a large number of UQ methods. <br>
**Built on PyTorch**: Native GPU & auto-diff support, seamless integration with deep learning pipelines. <br>
**Documentation**: Detailed tutorial to walk through popular UQ algorithms. Extensive documentation. <br>
**Extensive**: Supports calibration, conformal, multi-calibration, forecast evaluation, etc. 

### For researchers 

Torchuq provides a platform for conducting and distributing UQ research with the following key features:

**Baseline**: high quality implementation of many popular baseline methods to standardize comparison. <br>
**Benchmark**: a large set of datasets from recent UQ papers with a retrieved by a unified interface. <br> 
**Distribute your research**: you are welcome to distribute your algorithm via the TorchUQ interface. See [link]. 

## Installation 

First download the torchuq from pypi. To run the code, you can install the dependencies with the follwoing command

```
pip3 install requirements
```

pypi package link to come 

## Quickstart 

We first import torchuq and the functions that we will use. 
```
import torchuq
from torchuq.evaluate import distribution 
from torchuq.transform.conformal import ConformalCalibrator 
from torchuq.dataset import create_example_regression  
```
In this very simple example, we create a synthetic prediction (which is a set of Gaussian distributions) and recalibrate them with conformal calibration. 
```
predictions, labels = create_example_regression()
```
The example predictions are intentionally incorrect (i.e. the label is not drawn from the predictions). 
We will recalibrate the distribution with a powerful recalibration algorithm called conformal calibration. It takes as input the predictions and the labels, and learns a recalibration map that can be applied to new data (here for illustration purposes we apply it to the original data). 

```
calibrator = ConformalCalibrator(input_type='distribution', interpolation='linear')
calibrator.train(predictions, labels)
adjusted_predictions = calibrator(predictions)
```
We can plot these distribution predictions as a sequence of density functions, and the labels as the cross-shaped markers. 
As shown by the plot, the original predictions have systematically incorrect variance and mean, which is fixed by the recalibration algorithm. 

```
distribution.plot_density_sequence(predictions, labels, smooth_bw=10)
distribution.plot_density_sequence(adjusted_predictions, labels, smooth_bw=10)
```

![plot_original](docs/illustrations/quickstart_plot.svg)
![plot_calibrate](docs/illustrations/quickstart_plot2.svg)

## What's Next? 

A good way to start is to read about the [basic design philosophy and usage](https://torchuq.github.io/docs/overview.html), then go through these [tutorials](https://github.com/TorchUQ/torchuq/tree/main/examples/tutorial). All the tutorials are interactive jupyter notebooks. You can either download them to run locally or view them statically [here](https://torchuq.github.io/docs/tutorials/index.html). 

