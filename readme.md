TorchUQ is your one-stop solution for uncertainty quantification. At its core, TorchUQ supports various representations of uncertainty (including probability, particle/ensemble, conformal sets, etc), and 1. converts between different representations of uncertainty 2. adjusts any uncertainty representation so it becomes calibrated/valid 3. evaluates and visualizes the quality of uncertainty quantification. TorchUQ is built on pytorch, and supports auto-differentiation for most of its functions. Support for numpy interface to come. 


Why use torchuq to evaluate your model 

- The most comprehensive set of evaluation metrics 

- Illustrative tutorial explaining the different metrics 

- A one-size-fit-all evaluation function that automatically generates a comprehensive report of model performance.

- An extended set of benchmarks that are not available in uncertainty-benchmark, including imagenet superclasses 

Why use torchuq to recalibrate your model

- Simple unified interface for x different recalibration algorithms, including isotonic regression, histogram binning, dirichlet calibration, decision calibration, etc

- Most recalibration methods are implemented in pytorch and differentiable. This means that you can treat the calibrator function as a network layer. For advanced users you can even set Calibrator.differentiable = True and optimize the parameters of the recalibrator itself. You can know which calibration class is differentiable by querying the flag Calibrator.is_differentiable 

- Full GPU support, with all methods implemented with pytorch backend. A numpy interface is also provided, even though it is no longer differentiable. 

- Support for binary classification, multi-class classification and 1D regression 

- Pretrained calibrators for pytorch-vision models 


Several types of probabilities are accepted. These include 

Continuous distributions: any class that has a cdf and icdf method. This includes any if the pytorch Distribution classes. If this is the input to a calibrator, then the output is also a pytorch distribution class, but only the cdf, icdf methods are implemented; the mean and stddev fields are also available. If the cdf method of the original class is differentiable, then the cdf method of the new class is also differentiable. 

Categorical distributions: a probability vector. Binary classification is also supported, a probability vector of shape batch_size, 1 is automatically interpreted as a Bernoulli distribution. 


There are several types of representations for a probability distribution 

1. Categorical distribution: (taking values in $$\lbrace 0, 1, \cdots, K-1 \rbrace$$) as a 2-D array of floats `[batch_size, K]`, the number of possible values $$K$$ are automatically induced by the shape of the array. 

2. Bernoulli distribution: (taking values in $$\lbrace 0, 1 \rbrace$$) as a 1-D array of floats `[batch_size]`. It is also possible to represent a Bernoulli distribution as a categorical distribution.  

3. Empirical distribution: as a 2-D array of floats `[batch_size, n_samples]`. To distinguish empirical distributions from categorical distributions, all functions that can take as input an empirical distribution, also has an optional argument empirical_distribution=False/True. The default value is False (i.e. not an empirical distribution). 

4. Point prediction: Point predictions can also be represented as an empirical distribution with 1 sample. It could be a 1-D array of floats `[batch_size]` or a 2-D array of floats `[batch_size, n_sample]`

5. Arbitrary parametric distribution: as a python list of ``torch.distribution.Distributions`` instances. 



We support the following calibration algorithms: 

Binary calibration: Histogram binning, isotonic regression 

Multi-class calibration: confidence histogram binning, Dirichlet calibration, Decision calibration

Regression calibration: isotonic regression, conformal calibration 

Groupwise calibration: group-wise recalibration, multi-accuracy recalibration

