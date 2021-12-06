
# Basic Design Philosophy and Usage

The core of torchuq consists of the following three components: prediction types, evaluation and transform.

## 1. Prediction types 

Before we start to work with any predictions we must think about how to represent our prediction. For example, when predicting the price of a house, we could represent it as a range, such as 120k-150k dollars, or we could represent it as a cumulative density function (CDF). Torchuq supports around 10 different prediction types for regression and classification problems: 

- For a list of types supported for regression problems, see [link] 

- For a list of types supported for classification problems, see [link] 

- In the next major update, we intend to support additional prediction types, such as multi-variate predictions. 

## 2. Evaluation Metrics 

All evaluation metrics are in the torchuq.evaluate sub-module. There are two main types of evaluations: computing a metric or making a visualization plot. 

### 2.1 Computing a metric 

These functions compute a evaluation metric (such as L2 loss or ECE error) on a batch of predictions. All metric computation functions take the following format

``` torchuq.evaluate.{prediction_type}.compute_{metirc_name}(predictions, labels, reduction='mean')```

For example, to compute the ECE of a categorical prediction, use

```torchuq.evaluate.categorical.compute_ece(predictions, labels)```

Most metric evaluation functions take three arguments (but some may take more or less arguments)

- ```predictions```: the prediction that we would like to evaluate. Must have the correct type. For example, if we use a function in the module ```torchuq.evaluate.categorical``` then the prediction must have ```categorical``` type. 
- ```labels```: the true labels, not required for all functions. This should be an array of int for classification problems or an array of floats for regression problems.  
- ```reduction```: the str that decides how the computed metric is aggregated across the batch. This argument works in the same way as in pytorch. For example, the mean reduction indicates that we want to average the evaluation metrics. 

### 2.2 Make a plot

These functions make a plot to visualize the quality of the batch of predictions. 

``` torchuq.evaluate.{predition_type}.plot_{visualization_name}(predictions, labels, ax=None)```

For example, to compute the reliability diagram of a categorical prediction, use

```torchuq.evaluate.categorical.plot_reliability_diagram(predictions, labels)``` 

Most visualization functions take three arguments (but some may take more or less arguments)

- ```predictions, labels```: same as metric evaluation functions
- ```ax```: the matplotlib axes to make the figure on. If ```ax is None``` (recommended), then a new figure (of suitable size) will be created. If ```ax is not None``` then you should make sure the figure has the right size for visual appeal of the plot. 


## 3. Transform

### 3.1 Simple Transform

Depending on the different requirements during training/deployment, we might want to convert between different prediction types. For example, we might initially start from an ensemble prediction (maybe because we trained multiple prediction models), then convert it into a cumulative density function prediction or a point prediction (which are more interpretable and easier to work with). 

Torchuq supports conversion whenever the conversion makes sense. For details see [link] 

### 3.2 Calibrator

The calibrator class consists of three main functions. 

```Calibrator.__init__```(input_type='auto')

Input type can be any of the prediction types in the previous section. 

``` Calibrator.train(predictions, labels, side_feature=None) ```

In addition, because many recalibration algorithms (such as group calibration or multicalibration allow additional side features, we also allow an additional argument side_feature. Most recalibration algorithms do not use it, hence we do not consider it.)



```Calibrator.__call__(predictions, side_feature=None)```

For example, to use temperature scaling to recalibrate a categorical prediction use 

```
calibrator = TemperatureScaling(verbose=True)
calibrator.train(predictions, labels)
predictions_ts = calibrator(predictions)
```

Finally many algorithms can be used in the online prediction setup, where data becomes available in a sequential order and can be used to update the best predictor for future data. This is achieved by 

```Calibrator.update(predictions, labels, side_feature=None) ```

which works in the same way as ```calibrator.train``` but should update the calibrator with additional data. For an online prediction example see [link]. 
