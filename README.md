# Why TorchUQ 

TorchUQ is a one-stop solution for uncertainty quantification (UQ).

Accurate uncertainty quantification (UQ) is extremely important in high-stakes applications such as autonomous driving, healthcare, and public policy --- prediction models in such applications should know what they do not know. UQ also finds numerous applications in active learning, statistical inference, or in natural science and engineering applications that are rife with sources of uncertainty. 

However, there is no general consensus on how to accurately quantify uncertainty. For example, a common criterion for accurate UQ, calibration, has tens of different definitions, and each definition is suitable for some application but not others. This lack of consensus can confuse practitioners who might not know which notion to use, and can hinder UQ research because of the difficulty of implementing numerous baselines. TorchUQ aims to bridge this gap by 

1. summarizing and classifying different UQ methods;

2. defining a unified interface for using UQ methods, such that downstream users can **plug-and-play** different UQ methods with 1-2 lines of code;

3. providing **efficient and well-tested implementations** of common UQ methods, such as calibration, conformal prediction, multi-calibration, and evaluation/visualization techniques. 

In addition, TorchUQ has the following benefits:

4. TorchUQ is based on pytorch, so all native functionalities **support GPU** acceleration with no overhead (If a function receives GPU tensors as input, then it is automatically computed on GPU). Most functions are also **end-to-end differentiable** and can be incorporated into a deep learning pipeline.

5. TorchUQ includes a **large set of tutorials** to illustrate popular algorithms and evaluation metrics for UQ.

6. TorchUQ comes with a large set of **benchmark datasets** used in recent UQ papers with a one-line interface to retrieve these datasets.

# Installation 

First download the torchuq from pypi. To run the code, you can install the dependencies with the follwoing command

```
pip3 install requirements
```


# Quickstart 

```
import torch, torchuq
from torchuq.evaluate import distribution 
from torchuq.transform.conformal import ConformalCalibrator 
from torchuq.dataset import create_example_regression  
```
In this very simple example, we create a synthetic prediction (which is a set of Gaussian distributions) and recalibrate them with conformal calibration. `
```
predictions, labels = create_example_regression()
```
The example predictions are intentially incorrect (i.e. the label is not drawn from the predictions). 
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

A good way to start is to continue reading to learn about the basic design philosophy and usage of the package, then go through these [tutorials](https://github.com/TorchUQ/torchuq/tree/main/examples/tutorial). All the tutorials are interactive jupyter notebooks. You can either download them to run locally, use colab, or view them here. 


